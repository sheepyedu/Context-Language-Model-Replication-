"""Prepare the corrected episode admission without changing the first test bundle."""
import hashlib
import json
from pathlib import Path
import shutil
import time
root=Path('/mount-data/CLM-Replication')
prepared=root/'management/summary-harness-prepared-v2'
assert json.loads((prepared/'cpu-container-r3/EXIT.json').read_text())['exit_code']==1
bundle=root/'agent-bundle-summary-v4/clm_replication'
shutil.copytree(root/'agent-bundle-summary-v3/clm_replication',bundle,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['summary_loop.py','summary_runner.py','summary_polar_agent.py']:
    shutil.copy2(prepared/name,bundle/name)
files={str(path.relative_to(bundle)):hashlib.sha256(path.read_bytes()).hexdigest()
       for path in sorted(bundle.rglob('*')) if path.is_file() and '__pycache__' not in path.parts}
proof={'prepared':True,'created':time.time(),'bundle_sha256':files,'dependency_source':'ASSETS_PREPARED.json',
       'new_model_calls':0,'active_C2_source_changed':False,'reason':'token counter normalizes tool argument strings with the same valid-JSON object normalization as SGLang, using the existing Python standard library, preserving the wire messages'}
(prepared/'BUNDLE_R4_PREPARED.json').write_text(json.dumps(proof,indent=2))
print({'prepared':True,'files':len(files)})
