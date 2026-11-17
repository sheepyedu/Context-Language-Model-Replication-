"""Run one CPU-only data audit and retain the child's actual exit status."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True, type=Path)
    args, forwarded = parser.parse_known_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name('audit_update_data.py')
    identity = {
        'pid': os.getpid(), 'uid': os.getuid(),
        'start_ticks': int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'argv_sha256': hashlib.sha256(Path('/proc/self/cmdline').read_bytes()).hexdigest(),
    }
    command = [sys.executable, str(script), *forwarded,
               '--output', str(args.directory / 'result')]
    (args.directory / 'INTENT.json').write_text(json.dumps({
        'created': time.time(), 'owner': identity, 'command': command,
        'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
        'parent_wait_seconds': 330, 'new_model_tool_judge_calls': 0,
    }, indent=2))
    environment = os.environ.copy()
    environment.update({'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                        'MKL_NUM_THREADS': '1'})
    with (args.directory / 'stdout.raw').open('xb') as output, \
            (args.directory / 'stderr.raw').open('xb') as errors:
        child = subprocess.Popen(command, stdout=output, stderr=errors, env=environment)
        (args.directory / 'CHILD_LAUNCH.json').write_text(json.dumps({
            'pid': child.pid, 'created': time.time(),
        }))
        try:
            code = child.wait(timeout=330)
        except subprocess.TimeoutExpired:
            child.kill()
            code = child.wait()
    (args.directory / 'OWNER_EXIT.json').write_text(json.dumps({
        'child_actual_exit_code': code, 'finished': time.time(), 'owner': identity,
        'parameter_update_accepted': False, 'new_model_tool_judge_calls': 0,
    }, indent=2))
    return 0 if code == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
