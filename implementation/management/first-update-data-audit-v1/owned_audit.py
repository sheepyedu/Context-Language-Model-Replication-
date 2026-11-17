"""Collect the actual bounded CPU audit child exit status."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time


directory = Path(__file__).parent
environment = os.environ.copy()
environment['CUDA_VISIBLE_DEVICES'] = ''
environment['OMP_NUM_THREADS'] = '1'
environment['MKL_NUM_THREADS'] = '1'
with (directory / 'worker.log').open('xb') as log:
    worker = subprocess.Popen([sys.executable, str(directory / 'audit_first_update_data.py')],
                              stdout=log, stderr=subprocess.STDOUT, env=environment)
    (directory / 'WORKER_LAUNCH.json').write_text(json.dumps({'pid': worker.pid, 'created': time.time()}))
    try:
        code = worker.wait(timeout=330)
    except subprocess.TimeoutExpired:
        worker.kill()
        code = worker.wait()
    (directory / 'OWNER_EXIT.json').write_text(json.dumps({'exit_code': code, 'created': time.time()}))
sys.exit(code)
