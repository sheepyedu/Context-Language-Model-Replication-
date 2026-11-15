"""Drain the existing gateway, then replace it with a versioned judge."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

import yaml

root = Path('/mount-data/CLM-Replication')
output = root / 'management/judge-drain-r1'
with (output / 'HANDOFF_INTENT-v2.json').open('x') as handle:
    json.dump({'created': time.time(), 'preserve_training_controller': True,
               'preserve_sglang_service': True, 'replay_completed_calls': 0}, handle)
old = json.loads((root / 'management/gateway-archive-v5.launch.json').read_text())
pid = old['new_pid']
assert not (output / 'NEW_GATEWAY_LAUNCH.json').exists()
listeners = subprocess.check_output(['ss', '-lntp'], text=True)
assert any(':19171 ' in line and f'pid={pid},' in line for line in listeners.splitlines())
process_path = Path('/proc') / str(pid)
argv = (process_path / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
assert 'clm_replication.gateway' in argv and str(root / 'management/topology-v2.yaml') in argv
identity = {'pid': pid, 'argv': argv, 'start_ticks': int((process_path / 'stat').read_text().split()[21])}
(output / 'OLD_IDENTITY.json').write_text(json.dumps(identity, indent=2))

def get(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)

def fully_drained(sessions, metrics):
    stages = ('init_queue_depth', 'init_inflight', 'ready_depth', 'run_inflight',
              'postrun_queue_depth', 'postrun_inflight')
    return not sessions and all(metrics.get(stage) == 0 for stage in stages)


deadline = time.monotonic() + 1200
while True:
    health = get('http://127.0.0.1:19171/health')
    sessions = get('http://127.0.0.1:19171/sessions')['sessions']
    (output / 'LAST_DRAIN_OBSERVATION.json').write_text(json.dumps({
        'created': time.time(), 'sessions': sessions, 'metrics': health['metrics'],
    }, indent=2))
    if fully_drained(sessions, health['metrics']):
        break
    if time.monotonic() > deadline:
        raise TimeoutError('Original sessions have not finished within bounded handoff window')
    time.sleep(2)
status = get('http://127.0.0.1:19171/admin/inference/status')
assert status['inflight'] == 0
assert int((process_path / 'stat').read_text().split()[21]) == identity['start_ticks']
os.kill(pid, signal.SIGTERM)
for _ in range(80):
    if not process_path.exists():
        break
    time.sleep(0.25)
assert not process_path.exists(), 'Previous gateway has not exited'
with socket.socket() as connection:
    assert connection.connect_ex(('127.0.0.1', 19171)) != 0
revision = root / 'source-revisions/judge-no-thinking-v1'
topology = yaml.safe_load((root / 'management/topology-v2.yaml').read_text())
topology['gateway']['nodes'][0]['id'] = 'clm-gpu45-judge-v2'
config = root / 'management/topology-judge-v2.yaml'
config.write_text(yaml.safe_dump(topology, sort_keys=False))
python_path = ':'.join(str(p) for p in (revision, root / 'src', root / 'src/ProRL/src', root / 'src/slime', root / 'src/Megatron-LM'))
command = [str(root / 'management/runtime.sh'), 'env', f'PYTHONPATH={python_path}',
           'python', '-m', 'clm_replication.gateway', '--topology', str(config),
           '--node-id', 'clm-gpu45-judge-v2']
with (output / 'NEW_GATEWAY.log').open('wb') as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
receipt = {'old_identity': identity, 'old_gone': True, 'new_pid': process.pid,
           'argv': command, 'created': time.time(), 'source_revision': str(revision),
           'model_or_tool_replay': 0, 'main_training_restarted': False}
(output / 'NEW_GATEWAY_LAUNCH.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2), flush=True)
