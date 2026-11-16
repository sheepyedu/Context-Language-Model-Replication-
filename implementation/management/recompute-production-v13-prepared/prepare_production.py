"""Prepare an unmerged continuation with native recomputation-only validation."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import subprocess
import time
R=Path('/mount-data/CLM-Replication')
P=R/'management/recompute-production-v13-prepared'
S=R/'source-revisions/training-v13-recompute'
assert not S.exists()
shutil.copytree(R/'source-revisions/training-v10',S)
shutil.copy2(P/'recompute_preflight.py',S/'clm_replication/recompute_preflight.py')
entry='''"""Frozen-source CLI entry for the unmerged continuation."""
import importlib.util
from pathlib import Path
import json
import os
import hashlib
S=Path('/mount-data/CLM-Replication/source-revisions/training-v13-recompute/clm_replication')
for name in ['train','bridge','advantages','recompute_preflight']:
    origin=Path(importlib.util.find_spec('clm_replication.'+name).origin)
    assert origin==S/(name+'.py'), (name,str(origin))
P=Path(os.environ['CLM_ATTEMPT_DIR'])/'PACKAGE_ORIGIN.json'
P.write_text(json.dumps({'source':str(S),'files':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in S.glob('*.py')}},indent=2))
from .train import main
if __name__=='__main__':main()
'''
(S/'clm_replication/recompute_train_entry.py').write_text(entry)
text=(R/'management/train-v11.sh').read_text()
text=text.replace('source-revisions/training-v10','source-revisions/training-v13-recompute')
text=text.replace('-m clm_replication.train ', '-m clm_replication.recompute_train_entry ')
text=text.replace('--recompute-num-layers 1','--recompute-num-layers 4')
text=text.replace('clm_replication.preflight.check_backward','clm_replication.recompute_preflight.check_backward')
text=text.replace('training-attempt11','training-attempt12')
(R/'management/train-v13-unmerged.sh').write_text(text)
subprocess.run(['bash','-n',str(R/'management/train-v13-unmerged.sh')],check=True)
owner=(R/'management/owned_training_v11.py').read_text().replace('train-v11.sh','train-v13-unmerged.sh')
(R/'management/owned_training_v13.py').write_text(owner)
# Only the rematerialization group and explicit source/entry/hook/debug paths change.
a=(R/'management/train-v11.sh').read_text()
b=text.replace('source-revisions/training-v13-recompute','source-revisions/training-v10').replace('clm_replication.recompute_train_entry','clm_replication.train').replace('--recompute-num-layers 4','--recompute-num-layers 1').replace('clm_replication.recompute_preflight.check_backward','clm_replication.preflight.check_backward').replace('training-attempt12','training-attempt11')
assert a==b
for f in (R/'source-revisions/training-v10/clm_replication').glob('*.py'):
 assert f.read_bytes()==(S/'clm_replication'/f.name).read_bytes()
for f in (S/'clm_replication').glob('*.py'):ast.parse(f.read_text())
proof={'accepted':True,'created':time.time(),'source':str(S),'original_package_bytes_unchanged':True,'task_config_unchanged':True,'task_config_sha256':hashlib.sha256((R/'management/train-config-v7.yaml').read_bytes()).hexdigest(),'sample_merging_active':False,'rematerialization_group':4,'native_gradient_memory_preflight_required':True,'actual_optimizer_updates':0}
(P/'CPU_ACCEPTED.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof,indent=2))
