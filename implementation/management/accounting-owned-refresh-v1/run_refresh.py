"""Run one bounded read-only accounting snapshot and collect its actual exit."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path('/mount-data/CLM-Replication')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=False)
    script = ROOT / 'management/non-chat-rejection-accounting-v1/token_report_v5.py'
    identity = {'pid': os.getpid(), 'uid': os.getuid(),
                'start_ticks': int(Path('/proc/self/stat').read_text().split()[21]),
                'argv_sha256': hashlib.sha256(Path('/proc/self/cmdline').read_bytes()).hexdigest()}
    intent = {'created': time.time(), 'owner': identity, 'script': str(script),
              'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
              'child_alarm_seconds': 290, 'parent_wait_seconds': 310,
              'new_model_tool_judge_calls': 0}
    (directory / 'INTENT.json').write_text(json.dumps(intent, indent=2))
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(ROOT / 'management') + ':' + environment.get('PYTHONPATH', '')
    environment['CUDA_VISIBLE_DEVICES'] = ''
    child_code = None
    publication_code = 1
    with (directory / 'stdout.raw').open('xb') as output, (directory / 'stderr.raw').open('xb') as errors:
        child = subprocess.Popen([sys.executable, str(script)], stdout=output, stderr=errors,
                                 env=environment, preexec_fn=lambda: signal.alarm(290))
        (directory / 'CHILD_LAUNCH.json').write_text(json.dumps({'pid': child.pid, 'created': time.time()}))
        try:
            child_code = child.wait(timeout=310)
        except subprocess.TimeoutExpired:
            child.kill()
            child_code = child.wait()
    if child_code == 0:
        try:
            snapshot = json.loads((directory / 'stdout.raw').read_bytes())
            assert snapshot['all_dimensions_reconcile'] and snapshot['csv_all_roles_reconcile']
            temporary = directory / 'SNAPSHOT.tmp'
            temporary.write_text(json.dumps(snapshot, indent=2))
            temporary.replace(directory / 'SNAPSHOT.json')
            publication_code = 0
        except (ValueError, KeyError, AssertionError) as error:
            (directory / 'PUBLICATION_ERROR.json').write_text(json.dumps({
                'type': type(error).__name__, 'message': str(error)}, indent=2))
    (directory / 'EXIT.json').write_text(json.dumps({
        'created': time.time(), 'child_actual_exit_code': child_code,
        'publication_exit_code': publication_code, 'owner': identity,
        'previous_snapshots_modified': False, 'new_model_tool_judge_calls': 0,
    }, indent=2))
    return publication_code


if __name__ == '__main__':
    sys.exit(main())
