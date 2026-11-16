"""Exercise real process expiry/death against the running native admission code."""

from argparse import Namespace
import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from clm_replication.bridge import wait_for_task_admission


ROOT = Path('/mount-data/CLM-Replication/management')


def await_published(process, gate):
    deadline = time.monotonic() + 10
    while not gate.exists():
        if process.poll() is not None:
            raise RuntimeError('Lease owner exited before publication')
        if time.monotonic() >= deadline:
            raise TimeoutError('Lease publication')
        time.sleep(.02)


def main():
    directory = ROOT / 'admission-lease-regression-v1'
    directory.mkdir(exist_ok=False)
    audit = directory / 'native-admission-audit'
    audit.mkdir()
    gate = directory / 'test-gate.json'
    children = []
    rows = []

    def launch(name, duration):
        log = (directory / (name + '.log')).open('wb')
        process = subprocess.Popen([
            sys.executable, str(ROOT / 'bounded_admission_lease.py'),
            '--gate', str(gate), '--directory', str(directory / name),
            '--duration-seconds', str(duration),
        ], stdout=log, stderr=subprocess.STDOUT)
        log.close()
        children.append(process)
        return process

    def wait_native(position):
        started = time.monotonic()
        args = Namespace(clm_admission_gate_path=str(gate), clm_audit_dir=str(audit))
        asyncio.run(asyncio.wait_for(
            wait_for_task_admission(args, 900, 0, position), timeout=8))
        return time.monotonic() - started

    try:
        process = launch('expiry', 1)
        await_published(process, gate)
        assert gate.is_symlink()
        payload = json.loads(gate.read_bytes())
        assert payload['owner']['pid'] == process.pid
        elapsed = wait_native(0)
        assert process.wait(timeout=5) == 0 and not gate.exists()
        assert json.loads((directory / 'expiry/EXIT.json').read_bytes())['reason'] == 'expired'
        rows.append({'case': 'native_wait_resumes_on_alarm', 'accepted': True,
                     'native_wait_seconds': elapsed, 'native_exit': 0})

        process = launch('owner-killed', 60)
        await_published(process, gate)
        os.kill(process.pid, signal.SIGKILL)
        assert process.wait(timeout=5) == -9
        assert gate.is_symlink() and not gate.exists()
        elapsed = wait_native(1)
        assert elapsed < 1
        rows.append({'case': 'SIGKILL_closes_descriptor_and_native_wait',
                     'accepted': True, 'native_wait_seconds': elapsed, 'native_exit': -9})

        process = launch('replace-dangling', 1)
        await_published(process, gate)
        elapsed = wait_native(2)
        assert process.wait(timeout=5) == 0 and not gate.exists()
        rows.append({'case': 'new_lease_replaces_only_dangling_link',
                     'accepted': True, 'native_wait_seconds': elapsed, 'native_exit': 0})

        first = launch('live-owner', 3)
        await_published(first, gate)
        target = os.readlink(gate)
        second = launch('collision', 1)
        assert second.wait(timeout=5) == 1
        assert first.poll() is None and gate.exists() and os.readlink(gate) == target
        os.kill(first.pid, signal.SIGTERM)
        assert first.wait(timeout=5) == 0 and not gate.exists()
        rows.append({'case': 'collision_preserves_live_lease_and_explicit_release',
                     'accepted': True, 'collision_exit': 1, 'native_exit': 0})

        process = launch('invalid-duration', 3601)
        assert process.wait(timeout=5) == 1 and not gate.exists()
        rows.append({'case': 'excessive_duration_rejected_before_gate',
                     'accepted': True, 'native_exit': 1})
    finally:
        for process in children:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
    assert all(not Path('/proc', str(process.pid)).exists() for process in children)
    proof = {'accepted': True, 'created': time.time(), 'cases': rows,
             'all_test_owners_exactgone': True, 'active_source_changed': False,
             'native_wait_source_file': inspect.getsourcefile(wait_for_task_admission),
             'native_wait_source_sha256': hashlib.sha256(
                 inspect.getsource(wait_for_task_admission).encode()).hexdigest(),
             'new_model_tool_or_judge_calls': 0, 'production_gate_used': False,
             'compatibility': 'Current training-v7 native path-exists admission loop'}
    (directory / 'CPU_ACCEPTED.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
