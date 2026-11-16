"""Build the pinned exact-attention dependency in an isolated target."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time

root = Path('/mount-data/CLM-Replication')
directory = root / 'management/attention-fa283-build/r2'
directory.mkdir(exist_ok=False)
environment = dict(os.environ, FLASH_ATTENTION_FORCE_BUILD='TRUE',
                   FLASH_ATTN_CUDA_ARCHS='90', MAX_JOBS='16', NVCC_THREADS='1')
command = [str(root / 'management/runtime.sh'), 'python', '-m', 'pip', 'install',
           '--no-deps', '--no-build-isolation', '--no-index', '--target',
           str(root / 'env/attention-fa283'),
           '/tmp/clm-attention-fa283-20261116/flash_attn-2.8.3']
(directory / 'INTENT.json').write_text(json.dumps({'created': time.time(), 'argv': command,
                                                'new_generation_calls': 0}, indent=2))
with (directory / 'BUILD.log').open('wb') as log:
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                             env=environment, start_new_session=True)
    (directory / 'CHILD.json').write_text(json.dumps({'pid': child.pid, 'created': time.time()}))
    try:
        code = child.wait(timeout=1200)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        code = child.wait(timeout=20)
    (directory / 'EXIT.json').write_text(json.dumps({'exit_code': code, 'finished': time.time()}))
