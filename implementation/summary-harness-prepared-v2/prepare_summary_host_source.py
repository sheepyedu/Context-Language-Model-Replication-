"""Build a separate host plugin package without shadowing the running C2 source."""
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path('/mount-data/CLM-Replication')
p=ROOT/'management/summary-harness-prepared-v2'
assert json.loads((p/'cpu-container-r4/EXIT.json').read_text())['exit_code']==0
source=ROOT/'source-revisions/training-v10/clm_replication'
target=p/'summary-host-source-v4/clm_replication'
shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copy2(p/'summary_polar_agent.py',target/'summary_polar_agent.py')
files={str(f.relative_to(target)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(target.rglob('*')) if f.is_file()}
assert all(hashlib.sha256((source/name).read_bytes()).hexdigest()==sha for name,sha in files.items() if name!='summary_polar_agent.py')
(p/'HOST_SOURCE_PREPARED.json').write_text(json.dumps({'prepared':True,'files':files,'parent_source':'training-v10','production_source_changed':False,'new_models':0},indent=2))
print({'files':len(files),'isolated_host_source':str(target)})
