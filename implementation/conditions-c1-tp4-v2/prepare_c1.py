"""Refresh the pending task-only control to the accepted four-card baseline."""
import hashlib,json,subprocess,time
from pathlib import Path
import yaml
R=Path('/mount-data/CLM-Replication');P=R/'management/conditions-c1-tp4-v2'
base=(R/'management/train-v14.sh').read_text()
text=base.replace('RUN="$ROOT/runs/c2-seed1234"','RUN="$ROOT/runs/c1-seed1234"')
text=text.replace('--custom-config-path "$ROOT/management/train-config-v7.yaml"','--custom-config-path "$ROOT/management/conditions-c1-tp4-v2/train-config-c1.yaml"')
text=text.replace('env CUDA_VISIBLE_DEVICES=4,5,6,7 ','env CUDA_VISIBLE_DEVICES=4,5,6,7 POLAR_DUAL_CHANNEL_W_EFF=0 ')
text=text.replace('"FLA_TILELANG":"1"}', '"FLA_TILELANG":"1","POLAR_DUAL_CHANNEL_W_EFF":"0"}')
text=text.replace('training-attempt13','training-attempt01')
(P/'train-c1-tp4.sh').write_text(text)
config=yaml.safe_load((R/'management/train-config-v7.yaml').read_text())
config['clm_run_id']='c1-seed1234';config['clm_audit_dir']=str(R/'runs/c1-seed1234/audit')
config['polar_task_id_template']='clm-c1-eval-{rollout_id}-{sample.metadata.qid}'
config['clm_admission_gate_path']=str(P/'PAUSE_NEW_TASKS.json')
(P/'train-config-c1.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
subprocess.run(['bash','-n',str(P/'train-c1-tp4.sh')],check=True)
validation=P/'validation';validation.mkdir(exist_ok=True)
import os
env=dict(os.environ,CLM_VALIDATE_CONFIG='1',CLM_ATTEMPT_DIR=str(validation))
with (P/'CONFIG_VALIDATION.log').open('wb') as log:
 result=subprocess.run(['bash',str(P/'train-c1-tp4.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
assert result.returncode==0,result.returncode
args=json.loads((validation/'effective_args.json').read_text())
c2=json.loads((R/'runs/c2-seed1234/attempt-13/effective_args.json').read_text())
changed={key:{'c2':c2.get(key),'c1':args.get(key)} for key in c2.keys()|args.keys() if c2.get(key)!=args.get(key)}
expected={'custom_config_path','train_env_vars','save_debug_rollout_data','save','clm_run_id','save_debug_train_data','clm_admission_gate_path','clm_audit_dir','polar_task_id_template'}
assert changed.keys()==expected,changed
assert args['train_env_vars']=={**c2['train_env_vars'],'POLAR_DUAL_CHANNEL_W_EFF':'0'}
assert args['tensor_model_parallel_size']==4 and args['rollout_num_gpus_per_engine']==4
assert args['recompute_num_layers']==1 and args['rollout_batch_size']==8
assert args['n_samples_per_prompt']==32 and args['num_rollout']==70
assert args['custom_megatron_before_log_prob_hook_path'] is None
(P/'EFFECTIVE_DIFF.json').write_text(json.dumps(changed,indent=2))
proof={'accepted':True,'created':time.time(),'expected_changed_keys':sorted(expected),'TP':4,'physical_gpus':[4,5,6,7],'baseline_launcher_sha256':hashlib.sha256(base.encode()).hexdigest(),'baseline_source_pins':'f62a2b87571097ea4b254a896e84dbc24a2f2d2c9732e83745f75b337972fce4','native_config_parsed':True,'source_package':'training-v10','new_actor_model_or_judge_calls':0,'C1_training_dispatched':False,'actual_GPU_environment_pending':True}
(P/'CONFIG_ACCEPTED.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
