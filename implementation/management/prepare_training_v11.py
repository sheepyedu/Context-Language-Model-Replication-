"""Prepare a trainer-only TileLang backend, retaining the immutable workload."""
import hashlib
import json
from pathlib import Path
import py_compile
import subprocess
import time

root = Path('/mount-data/CLM-Replication')
m = root / 'management'
source = root / 'source-revisions/training-v10'
launcher = (m / 'train-v10.sh').read_text().replace('training-attempt10/', 'training-attempt11/')
trainer_path = ':'.join(map(str,[root/'env/tilelang-0115',root/'env/attention-fa283',source,root/'src',root/'src/ProRL/src',root/'src/slime',root/'src/Megatron-LM']))
env = json.dumps({'CUDA_DEVICE_MAX_CONNECTIONS':'1','PYTHONPATH':trainer_path,'FLA_TILELANG':'1'},separators=(',',':'))
old = '--train-env-vars \'{"CUDA_DEVICE_MAX_CONNECTIONS":"1"}\''
assert old in launcher
launcher = launcher.replace(old, "--train-env-vars '" + env + "'")
(m / 'train-v11.sh').write_text(launcher)
owner = (m/'owned_training_v10.py').read_text().replace('train-v10.sh','train-v11.sh')
(m/'owned_training_v11.py').write_text(owner)
launch = (m/'launch_training_v10.py').read_text().replace('parameters.attempt == 10','parameters.attempt == 11').replace('owned_training_v10.py','owned_training_v11.py')
proof_guard = 'assert json.loads((root / "management/gdn-tilelang-check/ACCEPTED.json").read_text())["accepted"]\n'
launch = launch.replace('assert parameters.attempt == 11\n','assert parameters.attempt == 11\n'+proof_guard)
(m/'launch_training_v11.py').write_text(launch)
for name in ['owned_training_v11.py','launch_training_v11.py']:
    py_compile.compile(str(m/name),doraise=True)
subprocess.run(['bash','-n',str(m/'train-v11.sh')],check=True)
previous = json.loads((m/'SOURCE_PINS.training-v10.json').read_text())
files = dict(previous['files'])
for old_name,new_name in [('train-v10.sh','train-v11.sh'),('owned_training_v10.py','owned_training_v11.py'),('launch_training_v10.py','launch_training_v11.py')]:
    files.pop(str(m/old_name))
    files[str(m/new_name)] = hashlib.sha256((m/new_name).read_bytes()).hexdigest()
# Freeze the compiler/runtime files actually imported by the trainer-only overlay.
for path in sorted((root/'env/tilelang-0115').rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
files[str(m/'check_gdn_tilelang.py')] = hashlib.sha256((m/'check_gdn_tilelang.py').read_bytes()).hexdigest()
value = {**previous,'created':time.time(),'source_revision':'training-v10-reused-trainer-tilelang-v11','files':files,
         'method':'TP4 and frozen source-v10 retained, trainer-only TileLang0.1.15/TVM-FFI0.1.12 avoids Hopper Triton known incorrect-gradient path; SGLang dependencies unchanged'}
(m/'SOURCE_PINS.training-v11.json').write_text(json.dumps(value,indent=2))
print(json.dumps({'prepared':True,'files':len(files),'pins_sha256':hashlib.sha256((m/'SOURCE_PINS.training-v11.json').read_bytes()).hexdigest()}))
