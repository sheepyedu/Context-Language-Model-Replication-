"""Admit one C1 owner after the preceding GPU owner has actually exited."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

ROOT = Path('/mount-data/CLM-Replication')
PREPARED = ROOT / 'management/c1-owned-entry-prepared-v1'
ALLOWED = {
    '4': 'GPU-101b4bf3-3f4e-5706-eef9-7e3d38ba4ea3',
    '5': 'GPU-c64fa322-ea6c-ed73-b81b-67bddfd2ad9e',
    '6': 'GPU-d75359c6-2336-10a8-5eb5-379d8c05d1d0',
    '7': 'GPU-0e09fed4-3dc8-937b-5d89-95ba16b5edbb',
}


def process_start(proc):
    try:
        stat = (proc / 'stat').read_text()
    except FileNotFoundError:
        return None
    return int(stat[stat.rfind(')') + 2:].split()[19])


def predecessor_status(root=ROOT, proc_root=Path('/proc')):
    pointer = json.loads((root / 'runs/c2-seed1234/CURRENT_OWNER.json').read_text())
    directory = Path(pointer['directory'])
    identity = json.loads((directory / 'OWNER_IDENTITY.json').read_text())
    assert identity['pid'] == pointer['pid']
    saved = identity['stat']
    expected_start = int(saved[saved.rfind(')') + 2:].split()[19])
    proc = proc_root / str(pointer['pid'])
    actual_start = process_start(proc)
    same_process_present = actual_start == expected_start
    exit_path = directory / 'EXIT.json'
    original_exit = json.loads(exit_path.read_text()) if exit_path.exists() else None
    closed = (original_exit is not None and type(original_exit.get('exit_code')) is int
              and not same_process_present)
    return {'ready': closed, 'pid': pointer['pid'], 'expected_start': expected_start,
            'actual_start': actual_start, 'same_process_present': same_process_present,
            'original_exit': original_exit, 'pointer': pointer}


def require_closed_predecessor(root=ROOT, proc_root=Path('/proc')):
    status = predecessor_status(root, proc_root)
    if not status['ready']:
        raise RuntimeError('The preceding C2 owner is still present or has no integer exit')
    return status


def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def admit(args, previous, root=ROOT):
    run = root / 'runs/c1-seed1234'
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'DISPATCH.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dispatch(args, previous, run)


def dispatch(args, previous, run):
    if run.exists():
        for attempt in run.glob('attempt-*'):
            if not (attempt / 'DISPATCH_INTENT.json').exists():
                continue
            original_exit = json.loads((attempt / 'EXIT.json').read_text())
            assert type(original_exit['exit_code']) is int
            launch = json.loads((attempt / 'LAUNCH.json').read_text())
            saved = json.loads((attempt / 'OWNER_IDENTITY.json').read_text())['stat']
            expected = int(saved[saved.rfind(')') + 2:].split()[19])
            assert process_start(Path('/proc') / str(launch['pid'])) != expected
    pins_path = PREPARED / 'SOURCE_PINS.json'
    pins = json.loads(pins_path.read_text())
    for name, expected in pins['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    assert json.loads((PREPARED / 'CPU_ACCEPTED.json').read_text())['accepted']
    baseline_pins = ROOT / 'management/SOURCE_PINS.training-v14.json'
    assert hashlib.sha256(baseline_pins.read_bytes()).hexdigest() == pins['baseline_source_pins_sha256']
    for name, expected in json.loads(baseline_pins.read_text())['files'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    identities = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
    actual = dict(tuple(value.strip() for value in line.split(',')) for line in identities.splitlines())
    assert all(actual[index] == uuid for index, uuid in ALLOWED.items())
    applications = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], text=True)
    assert not any(uuid in applications for uuid in ALLOWED.values()), 'Authorized GPUs are occupied'
    with socket.socket() as connection:
        connection.settimeout(2)
        assert connection.connect_ex(('127.0.0.1', 19174)) != 0
    gateway = get_json('http://127.0.0.1:19171/admin/inference/status')
    coordinator = get_json('http://127.0.0.1:19170/rollout/status')
    sessions = get_json('http://127.0.0.1:19171/sessions')
    from polar.gateway.session import SessionStatus
    terminal = {status.value for status in SessionStatus.terminal()}
    assert gateway['inflight'] == 0
    assert coordinator['pipeline']['pending_sessions'] == 0
    assert all(session['status'] in terminal for session in sessions['sessions'])
    ray = json.loads((ROOT / 'management/unmerged-baseline-v14/RAY_ACCEPTED.json').read_text())
    assert sorted(item['visible'] for item in ray['GPU_mappings']) == ['4', '5', '6', '7']
    assert ray['session_path'] == str(Path('/tmp/clm-ray-unmerged-v13-20261116/session_latest').resolve())
    with socket.socket() as connection:
        connection.settimeout(2)
        assert connection.connect_ex(('127.0.0.1', 29477)) == 0
    assert require_closed_predecessor()['pointer'] == previous['pointer']
    directory = run / f'attempt-{args.attempt:02d}'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'DISPATCH_INTENT.json').open('x') as handle:
        json.dump({'created': time.time(), 'condition': 'C1', 'physical_gpus': ALLOWED,
                   'source_pins_sha256': hashlib.sha256(pins_path.read_bytes()).hexdigest(),
                   'predecessor': previous, 'initial_checkpoint': str(ROOT / 'checkpoints/qwen35-9b-initial')}, handle, indent=2)
    argv = [str(ROOT / 'env/rl/bin/python'), str(PREPARED / 'owned_c1.py'), '--attempt', str(args.attempt)]
    with (directory / 'controller.log').open('wb') as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launch = {'pid': process.pid, 'created': time.time(), 'argv': argv, 'attempt': args.attempt}
    (directory / 'LAUNCH.json').write_text(json.dumps(launch, indent=2))
    temporary = run / 'CURRENT_OWNER.tmp'
    temporary.write_text(json.dumps({**launch, 'directory': str(directory)}, indent=2))
    temporary.replace(run / 'CURRENT_OWNER.json')
    print(json.dumps(launch, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--inspect-predecessor', action='store_true')
    args = parser.parse_args()
    if args.inspect_predecessor:
        print(json.dumps(predecessor_status(), indent=2))
        return
    previous = require_closed_predecessor()
    assert args.attempt >= 1
    admit(args, previous)


if __name__ == '__main__':
    main()
