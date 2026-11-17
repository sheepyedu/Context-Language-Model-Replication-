"""Exercise the real audit and owned CLI using synthetic CPU input files."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from audit_update_data import native_identity
from check_batch_cpu import fixture


here = Path(__file__).resolve().parent
owner = native_identity(3060432)
assert owner['start_ticks'] == 51919204
assert owner['raw_sha256'] == '336761ab09adce6296a5fd5d20491a4c1fc9b37ad46a47f9f3fb3f33ae09013e'
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    run = root / 'input'
    run.mkdir()
    rollout, saved = fixture(1)
    torch.save(rollout, run / 'rollout-1.pt')
    torch.save(saved, run / 'train-1.pt')
    arguments = ['--run', str(run), '--rollout-id', '1',
                 '--owner-pid', str(owner['pid']), '--owner-start', str(owner['start_ticks']),
                 '--owner-raw-sha256', owner['raw_sha256']]
    output = root / 'first-result'
    command = [sys.executable, str(here / 'audit_update_data.py'),
               *arguments, '--output', str(output)]
    first = subprocess.run(command, text=True, capture_output=True, timeout=15)
    assert first.returncode == 0, first.stderr
    proof = json.loads((output / 'ACCEPTED.json').read_text())
    assert proof['accepted'] and proof['counts']['samples'] == 256
    assert not proof['parameter_update_accepted']
    original_sha = hashlib.sha256((output / 'ACCEPTED.json').read_bytes()).hexdigest()
    duplicate = subprocess.run(command, text=True, capture_output=True, timeout=15)
    assert duplicate.returncode == 1
    assert hashlib.sha256((output / 'ACCEPTED.json').read_bytes()).hexdigest() == original_sha

    wrong = list(arguments)
    wrong[wrong.index('--owner-start') + 1] = str(owner['start_ticks'] + 1)
    rejected = subprocess.run([sys.executable, str(here / 'audit_update_data.py'),
                               *wrong, '--output', str(root / 'wrong-owner')],
                              text=True, capture_output=True, timeout=15)
    assert rejected.returncode == 1
    assert not json.loads((root / 'wrong-owner/FAILURE.json').read_text())['accepted']

    owned = root / 'owned'
    collected = subprocess.run([sys.executable, str(here / 'owned_audit.py'),
                                '--directory', str(owned), *arguments],
                               text=True, capture_output=True, timeout=15)
    assert collected.returncode == 0, collected.stderr
    assert json.loads((owned / 'OWNER_EXIT.json').read_text())['child_actual_exit_code'] == 0
    assert json.loads((owned / 'result/ACCEPTED.json').read_text())['accepted']
    child = json.loads((owned / 'CHILD_LAUNCH.json').read_text())['pid']
    assert not (Path('/proc') / str(child)).exists()

assert native_identity(owner['pid']) == owner
result = {'accepted': True, 'checks': 4, 'scope': 'synthetic CPU CLI input only',
          'actual_native_buffer_audit': False, 'scientific_owner_unchanged': True,
          'new_model_tool_judge_calls': 0, 'cuda_initialized': torch.cuda.is_initialized()}
here.joinpath('CLI_CPU_ACCEPTED.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
