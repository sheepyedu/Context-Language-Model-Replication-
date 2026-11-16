"""Keep the real controller exit status for recovery and final acceptance."""

import json
import os
from pathlib import Path
import subprocess
import time
import argparse
import urllib.request


root = Path('/mount-data/CLM-Replication')
parser = argparse.ArgumentParser()
parser.add_argument('--attempt', type=int, required=True)
parameters = parser.parse_args()
directory = root / f'runs/c2-seed1234/attempt-{parameters.attempt:02d}'
(directory / 'OWNER_IDENTITY.json').write_text(json.dumps({
    'pid': os.getpid(), 'stat': Path('/proc/self/stat').read_text(),
    'cmdline': Path('/proc/self/cmdline').read_bytes().replace(b'\0', b' ').decode(),
}, indent=2))
router_argv = [str(root / 'management/runtime.sh'), 'python', '-m',
               'sglang_router.launch_router', '--host', '127.0.0.1', '--port', '19174',
               '--disable-health-check', '--log-level', 'warn', '--prometheus-port', '19175']
with (directory / 'router.log').open('wb') as router_log:
    router = subprocess.Popen(router_argv, stdout=router_log, stderr=subprocess.STDOUT)
(directory / 'ROUTER_LAUNCH.json').write_text(json.dumps({
    'pid': router.pid, 'argv': router_argv, 'created': time.time(),
}, indent=2))
code = 1
process = None
try:
    deadline = time.monotonic() + 40
    while True:
        if router.poll() is not None:
            raise RuntimeError('Owned inference router exited during startup')
        try:
            with urllib.request.urlopen('http://127.0.0.1:19174/workers', timeout=1) as response:
                assert response.status == 200
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError('Owned inference router did not become ready')
            time.sleep(0.5)
    os.environ['CLM_ATTEMPT_DIR'] = str(directory)
    process = subprocess.Popen(['bash', str(root / 'management/train-v11.sh')])
    (directory / 'CHILD_LAUNCH.json').write_text(json.dumps({'pid': process.pid, 'created': time.time()}))
    code = process.wait()
finally:
    router.terminate()
    try:
        router.wait(timeout=15)
    except subprocess.TimeoutExpired:
        router.kill()
        router.wait()
    (directory / 'ROUTER_EXIT.json').write_text(json.dumps({'pid': router.pid, 'exit_code': router.returncode}))
    (directory / 'EXIT.json').write_text(json.dumps({
        'exit_code': code, 'finished': time.time(), 'pid': process.pid if process else None,
    }))
raise SystemExit(code)
