"""Restore the previously accepted unmerged trainer without optional optimization gates."""
import hashlib,json,os,subprocess,time
from pathlib import Path
R=Path('/mount-data/CLM-Replication');P=R/'management/unmerged-baseline-v14'
old=R/'runs/c2-seed1234/attempt-12'
assert json.loads((old/'EXIT.json').read_text())['exit_code']==1
assert not Path('/proc',str(json.loads((old/'LAUNCH.json').read_text())['pid'])).exists()
rejected=[]
for rank in range(4):
 f=old/f'recompute-preflight-rank{rank}/RESULT.json';d=json.loads(f.read_text())
 assert not d['accepted'] and d['parameter_sha256_before']==d['parameter_sha256_after']
 assert d['optimizer_updates']==0 and d['RNG_restored'] and d['gradients_cleared']
 rejected.append({'rank':rank,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'failed_tensors':sum(not x['accepted'] for x in d['gradients']),'max_gradient_abs_diff':max(x['max_abs'] for x in d['gradients']),'case_results':d['cases']})
carry=json.loads((R/'management/BACKWARD_PREFLIGHT_V11_ROOT_ACCEPTED.json').read_text());assert carry['accepted']
a=(R/'management/train-v11.sh').read_text();b=(P/'train-v14.sh').read_text()
assert b.replace('training-attempt13','training-attempt11').replace('--log-probs-chunk-size 512 \\','--log-probs-chunk-size 512 --custom-megatron-before-log-prob-hook-path clm_replication.preflight.check_backward \\')==a
# No optimizer/data/kernel setting changed. Only already-accepted diagnostic hook is removed.
assert '--recompute-num-layers 1' in b and 'strict_training' not in b
for name in ['train-v14.sh','owned_training_v14.py','launch_training_v14.py']:
 (R/'management'/name).write_bytes((P/name).read_bytes())
subprocess.run(['bash','-n',str(R/'management/train-v14.sh')],check=True)
proof={'accepted':True,'created':time.time(),'scientific_source':'training-v10','baseline_implementation':'training-v11 launcher plus new debug paths, accepted preflight carried','optional_merge_active':False,'optional_checkpoint_group_change_active':False,'recompute_num_layers':1,'rejected_attempt12':rejected,'original_preflight':carry,'new_generation_calls':0,'optimizer_updates':0,'no_regular_file_pause_gate':True}
(P/'BASELINE_RESTORE_ACCEPTED.json').write_text(json.dumps(proof,indent=2))
files=dict(json.loads((R/'management/SOURCE_PINS.training-v11.json').read_text())['files'])
for f in list(files):
 if f.endswith(('train-v11.sh','owned_training_v11.py','launch_training_v11.py')):files.pop(f)
for f in list(P.glob('*.py'))+[R/'management'/n for n in ['train-v14.sh','owned_training_v14.py','launch_training_v14.py']]:files[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
for f,h in files.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
pins={'created':time.time(),'source_revision':'training-v14-unmerged-baseline','source_package':'training-v10','parent':'training-v11','files':files,'optional_optimizations_active':False}
f=R/'management/SOURCE_PINS.training-v14.json';f.write_text(json.dumps(pins,indent=2))
proof={'accepted':True,'pins_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'files':len(files),'rejected_attempt12_preserved':True,'original_largest_backward_preflight_reused':True}
(P/'PREPARED.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
