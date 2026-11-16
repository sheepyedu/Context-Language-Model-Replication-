"""Capture one bounded SSH observation and publish only complete JSON."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
COMMAND = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'h20-direct',
           '/mount-data/CLM-Replication/env/rl/bin/python',
           '/mount-data/CLM-Replication/management/monitor_training_current_v2.py']


def capture(root=ROOT):
    records = root / 'observations'
    records.mkdir(exist_ok=True)
    prefix = records / f'transport-{time.time_ns()}'
    try:
        result = subprocess.run(COMMAND, capture_output=True, timeout=45)
    except subprocess.TimeoutExpired as error:
        prefix.with_suffix('.stdout.raw').write_bytes(error.stdout or b'')
        prefix.with_suffix('.stderr.raw').write_bytes(error.stderr or b'')
        prefix.with_suffix('.json').write_text(json.dumps({'timed_out': True, 'integer_exit': None}))
        raise
    prefix.with_suffix('.stdout.raw').write_bytes(result.stdout)
    prefix.with_suffix('.stderr.raw').write_bytes(result.stderr)
    prefix.with_suffix('.json').write_text(json.dumps({'timed_out': False,
                                                     'integer_exit': result.returncode}))
    result.check_returncode()
    observation = json.loads(result.stdout)
    assert isinstance(observation, dict)
    assert all(key in observation for key in ['observed', 'owner', 'actual_owner', 'native_progress'])
    temporary = prefix.with_suffix('.validated.tmp')
    temporary.write_bytes(result.stdout)
    temporary.replace(root / 'OBSERVATION_LATEST.json')
    return observation


if __name__ == '__main__':
    print(json.dumps(capture(), ensure_ascii=False, indent=2))
