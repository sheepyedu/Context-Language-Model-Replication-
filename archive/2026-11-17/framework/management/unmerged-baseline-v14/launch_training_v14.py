"""Single-owner admission for the first C2 run on authorized physical GPUs."""

import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
import argparse


root = Path('/mount-data/CLM-Replication')
parser = argparse.ArgumentParser()
parser.add_argument('--attempt', type=int, required=True)
parser.add_argument('--pins-version', required=True)
parameters = parser.parse_args()
run_root = root / 'runs/c2-seed1234'
assert json.loads((run_root / 'EXIT.json').read_text())['exit_code'] == 1
assert not (Path('/proc') / str(json.loads((run_root / 'LAUNCH.json').read_text())['pid'])).exists()
for previous in run_root.glob('attempt-*'):
    if (previous / 'DISPATCH_INTENT.json').exists():
        assert (previous / 'EXIT.json').exists(), f'Unclosed previous owner at {previous}'
        assert not (Path('/proc') / str(json.loads((previous / 'LAUNCH.json').read_text())['pid'])).exists()
assert parameters.attempt == 13
assert json.loads((root / "management/gdn-tilelang-check/ACCEPTED.json").read_text())["accepted"]
assert json.loads((root / "management/FLASH_NATIVE_ROOT_ACCEPTED.json").read_text())["accepted"]
assert json.loads((root / "management/CACHED_FIRST_BATCH_ROOT_ACCEPTED.json").read_text())["accepted"]
assert json.loads((root/'management/unmerged-baseline-v14/BASELINE_RESTORE_ACCEPTED.json').read_text())['accepted']
directory = run_root / f'attempt-{parameters.attempt:02d}'
directory.mkdir(parents=True, exist_ok=True)
pins_path = root / f'management/SOURCE_PINS.{parameters.pins_version}.json'
pins = json.loads(pins_path.read_text())
for name, expected in pins['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
identities = subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits',
], text=True)
allowed = {
    '4': 'GPU-101b4bf3-3f4e-5706-eef9-7e3d38ba4ea3',
    '5': 'GPU-c64fa322-ea6c-ed73-b81b-67bddfd2ad9e',
    '6': 'GPU-d75359c6-2336-10a8-5eb5-379d8c05d1d0',
    '7': 'GPU-0e09fed4-3dc8-937b-5d89-95ba16b5edbb',
}
actual = dict(tuple(value.strip() for value in line.split(',')) for line in identities.splitlines())
assert all(actual[index] == uuid for index, uuid in allowed.items())
applications = subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits',
], text=True)
assert not any(uuid in applications for uuid in allowed.values()), 'Authorized cards are occupied'
with socket.socket() as connection:
    connection.settimeout(2)
    assert connection.connect_ex(('127.0.0.1', 19174)) != 0, 'Old initial-policy server is still listening'
status = json.load(urllib.request.urlopen('http://127.0.0.1:19171/admin/inference/status', timeout=5))
assert status['inflight'] == 0
ray_proof = json.loads((root / 'management/unmerged-baseline-v14/RAY_ACCEPTED.json').read_text())
assert sorted(item['visible'] for item in ray_proof['GPU_mappings']) == ['4', '5', '6', '7']
assert ray_proof['session_path'] == str(Path('/tmp/clm-ray-unmerged-v13-20261116/session_latest').resolve())
with (directory / 'DISPATCH_INTENT.json').open('x') as handle:
    json.dump({'created': time.time(), 'source_pins_sha256': hashlib.sha256(pins_path.read_bytes()).hexdigest(),
               'physical_gpus': allowed, 'initial_checkpoint': str(root / 'checkpoints/qwen35-9b-initial')},
              handle, indent=2)
with (directory / 'controller.log').open('wb') as log:
    process = subprocess.Popen([str(root / 'env/rl/bin/python'), str(root / 'management/owned_training_v14.py'),
                                '--attempt', str(parameters.attempt)],
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
receipt = {'pid': process.pid, 'created': time.time(),
           'argv': [str(root / 'env/rl/bin/python'), str(root / 'management/owned_training_v14.py'),
                    '--attempt', str(parameters.attempt)], 'attempt': parameters.attempt}
(directory / 'LAUNCH.json').write_text(json.dumps(receipt, indent=2))
pointer = run_root / 'CURRENT_OWNER.tmp'
pointer.write_text(json.dumps({**receipt, 'directory': str(directory)}, indent=2))
pointer.replace(run_root / 'CURRENT_OWNER.json')
print(receipt)
