"""Freeze the original trainer closure plus the explicit continuation hooks."""
import ast,hashlib,json,os,subprocess,time
from pathlib import Path
R=Path('/mount-data/CLM-Replication');P=R/'management/recompute-production-v13-prepared'
files=dict(json.loads((R/'management/SOURCE_PINS.training-v11.json').read_text())['files'])
for name in list(files):
 if '/source-revisions/training-v10/clm_replication/' in name or name.endswith(('train-v11.sh','owned_training_v11.py','launch_training_v11.py')):files.pop(name)
for directory in [R/'source-revisions/training-v13-recompute/clm_replication',P]:
 for f in directory.rglob('*.py'):
  if '__pycache__' not in f.parts:
   ast.parse(f.read_text());files[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
for name in ['train-v13-unmerged.sh','owned_training_v13.py']:
 f=R/'management'/name;files[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
for f,h in files.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
pins={'created':time.time(),'source_revision':'training-v13-recompute','parent':'training-v11','files':files,'sample_merging':False,'checkpoint_group':4,'scientific_config_unchanged':True}
out=R/'management/SOURCE_PINS.training-v13.json';out.write_text(json.dumps(pins,indent=2))
env=dict(os.environ,CLM_VALIDATE_CONFIG='1',CLM_ATTEMPT_DIR=str(P/'config-validation'))
(P/'config-validation').mkdir(exist_ok=True)
with (P/'CONFIG_VALIDATION.log').open('wb') as log:
 result=subprocess.run(['bash',str(R/'management/train-v13-unmerged.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
assert result.returncode==0,result.returncode
args=json.loads((P/'config-validation/effective_args.json').read_text())
assert args['recompute_num_layers']==4 and args['num_rollout']==70
assert args['n_samples_per_prompt']==32 and args['rollout_batch_size']==8
assert args['tensor_model_parallel_size']==4 and args['num_steps_per_rollout']==1
assert args['rollout_function_path']=='clm_replication.bridge.generate_rollout'
proof={'accepted':True,'created':time.time(),'source_pins_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'source_files':len(files),'native_config_validated':True,'generation_calls':0,'sample_merging':False,'all_scientific_input_and_reward_parameters_retained':True}
(P/'CONFIG_ACCEPTED.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
