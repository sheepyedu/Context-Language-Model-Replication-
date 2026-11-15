"""Keep the coordinator's real child exit and fixed-source identity."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/mount-data/CLM-Replication')
directory=root/'management/queue-timeout-recovery-v2'
pins=json.loads((directory/'SOURCE_PINS.json').read_bytes())
for name,expected in pins['files'].items():
    assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==expected
for name,expected in pins['external_files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected
(directory/'OWNER_IDENTITY.json').write_text(json.dumps({'pid':os.getpid(),'created':time.time(),
    'stat':Path('/proc/self/stat').read_text(),'argv':Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode()},indent=2))
command=[str(root/'management/runtime.sh'),'env','CUDA_VISIBLE_DEVICES=', 'python',
         str(directory/'terminal_coordinator.py'),'--topology',str(root/'management/topology-training-v7.yaml'),
         '--recovery-directory',str(directory/'snapshot')]
with (directory/'coordinator.log').open('wb') as handle:
    child=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT)
(directory/'CHILD_LAUNCH.json').write_text(json.dumps({'pid':child.pid,'created':time.time(),'argv':command},indent=2))
code=child.wait()
(directory/'EXIT.json').write_text(json.dumps({'pid':child.pid,'finished':time.time(),'exit_code':code},indent=2))
raise SystemExit(code)
