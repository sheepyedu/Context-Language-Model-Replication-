"""Resume the same candidate journal after a fully drained, scoped handoff."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

import psutil
import yaml

ROOT = Path('/mount-data/CLM-Replication')
OUTPUT = ROOT / 'management/training-v5-recovery'


def get(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def post(url, payload=None, method='POST'):
    request = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else b'',
                                     headers={'Content-Type': 'application/json'}, method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def fully_drained(health, sessions):
    fields = ('init_queue_depth', 'init_inflight', 'ready_depth', 'run_inflight',
              'postrun_queue_depth', 'postrun_inflight')
    return not sessions and all(health['metrics'].get(field) == 0 for field in fields)


def main():
    assert json.loads((OUTPUT/'NATIVE_FAILURE_CPU.json').read_text())['passed'] == 4
    assert 'CLM_CONFIG_VALIDATED' in (ROOT/'management/config-validation-v5-r2.log').read_text()
    with (OUTPUT/'HANDOFF_INTENT.json').open('x') as handle:
        json.dump({'created': time.time(), 'old_attempt': 4, 'new_attempt': 5,
                   'candidate_journal_preserved': True, 'initial_policy_optimizer_updates': 0}, handle)
    old_gateway = 2446676
    proc = Path('/proc')/str(old_gateway)
    identity = {'pid': old_gateway, 'start_ticks': int((proc/'stat').read_text().split()[21]),
                'argv': (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()}
    assert identity['start_ticks'] == 45275097 and 'topology-judge-v2.yaml' in identity['argv']
    node = get('http://127.0.0.1:19170/nodes/clm-gpu45-judge-v2')
    if not node['draining']:
        post('http://127.0.0.1:19170/nodes/clm-gpu45-judge-v2', method='DELETE')
    deadline = time.monotonic()+1800
    while True:
        health = get('http://127.0.0.1:19171/health')
        sessions = get('http://127.0.0.1:19171/sessions')['sessions']
        (OUTPUT/'LAST_DRAIN_OBSERVATION.json').write_text(json.dumps({
            'created': time.time(), 'metrics': health['metrics'], 'sessions': sessions,
        }, indent=2))
        if fully_drained(health, sessions):
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Still draining original task executions')
        time.sleep(2)
    assert get('http://127.0.0.1:19171/admin/inference/status')['inflight'] == 0
    assert not (ROOT/'runs/c2-seed1234/audit/sample-summary-policy-00000.json').exists()
    assert int((proc/'stat').read_text().split()[21]) == identity['start_ticks']
    os.kill(old_gateway, signal.SIGTERM)
    for _ in range(120):
        if not proc.exists():
            break
        time.sleep(0.25)
    assert not proc.exists()
    current = json.loads((ROOT/'runs/c2-seed1234/CURRENT_OWNER.json').read_text())
    assert current['attempt'] == 4 and current['pid'] == 2421832
    directory = Path(current['directory'])
    child = json.loads((directory/'CHILD_LAUNCH.json').read_text())['pid']
    drivers = [p for p in psutil.Process(child).children(recursive=True)
               if 'clm_replication.train' in p.cmdline()]
    assert len(drivers) == 1
    driver = drivers[0]
    retirement = {'created': time.time(), 'pid': driver.pid,
                  'start_ticks': int((Path('/proc')/str(driver.pid)/'stat').read_text().split()[21]),
                  'argv': driver.cmdline(), 'generation_inflight': 0, 'optimizer_updates': 0,
                  'candidate_journal': str(ROOT/'runs/c2-seed1234/audit')}
    (OUTPUT/'KNOWN_IDLE_RETIREMENT.json').write_text(json.dumps(retirement, indent=2))
    os.kill(driver.pid, signal.SIGTERM)
    deadline = time.monotonic()+150
    while not (directory/'EXIT.json').exists() or Path('/proc/2421832').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Previous owned controller has not closed')
        time.sleep(1)
    for _ in range(120):
        with socket.socket() as connection:
            closed = connection.connect_ex(('127.0.0.1', 19174)) != 0
        applications = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits',
        ], text=True)
        free = not any(uuid in applications for uuid in (
            'GPU-101b4bf3-3f4e-5706-eef9-7e3d38ba4ea3', 'GPU-c64fa322-ea6c-ed73-b81b-67bddfd2ad9e'))
        if closed and free:
            break
        time.sleep(1)
    assert closed and free
    topology = yaml.safe_load((ROOT/'management/topology-judge-v2.yaml').read_text())
    topology['gateway']['nodes'][0]['id'] = 'clm-gpu45-training-v5'
    topology_path = ROOT/'management/topology-training-v5.yaml'
    topology_path.write_text(yaml.safe_dump(topology, sort_keys=False))
    prefix = ':'.join(str(p) for p in (ROOT/'source-revisions/training-v5', ROOT/'src',
                                      ROOT/'src/ProRL/src', ROOT/'src/slime', ROOT/'src/Megatron-LM'))
    command = [str(ROOT/'management/runtime.sh'), 'env', f'PYTHONPATH={prefix}',
               'CLM_GATEWAY_START_PAUSED=1', 'python', '-m', 'clm_replication.gateway',
               '--topology', str(topology_path), '--node-id', 'clm-gpu45-training-v5']
    with (OUTPUT/'NEW_GATEWAY.log').open('wb') as log:
        gateway = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (OUTPUT/'NEW_GATEWAY_LAUNCH.json').write_text(json.dumps({
        'pid': gateway.pid, 'argv': command, 'created': time.time(), 'old_identity': identity,
    }, indent=2))
    deadline = time.monotonic()+45
    while True:
        try:
            state = get('http://127.0.0.1:19171/admin/inference/status')
            assert state['paused'] is True and state['inflight'] == 0
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)
    (OUTPUT/'PAUSED_START_ACTUAL.json').write_text(json.dumps(state, indent=2))
    launch = subprocess.run([str(ROOT/'management/runtime.sh'), 'python',
                             str(ROOT/'management/launch_training_v5.py'), '--attempt', '5',
                             '--pins-version', 'training-v5-r2'], capture_output=True, text=True)
    (OUTPUT/'DISPATCH_RECEIPT.json').write_text(json.dumps({
        'exit_code': launch.returncode, 'stdout': launch.stdout, 'stderr': launch.stderr,
    }, indent=2))
    assert launch.returncode == 0, launch.stderr
    print(launch.stdout, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        (OUTPUT/'HANDOFF_ERROR.json').write_text(json.dumps({
            'type': type(error).__name__, 'message': str(error), 'created': time.time(),
        }, indent=2))
        raise
