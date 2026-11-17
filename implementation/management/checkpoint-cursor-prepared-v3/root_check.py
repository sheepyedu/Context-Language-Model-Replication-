import pathlib,json,hashlib,time,ast
r=pathlib.Path('/mount-data/CLM-Replication');p=r/'management/checkpoint-cursor-prepared-v3';sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
s=json.loads((p/'SOURCE_PINS.json').read_text())
for f,h in s['files'].items():assert sha(p/f)==h,f
assert sha(r/'source-revisions/training-v10/clm_replication/bridge.py')=='b68579dd2140ee8855db9859654b9fa734a1fdc0dec77894dc6661278f1e87df'
ray=json.loads((p/'RAY_CPU_ACCEPTED.json').read_text());assert ray['accepted'] and ray['actual_CPU_actor_exactgone'];assert not pathlib.Path('/proc',str(ray['actual_CPU_Ray_copy_and_hook']['pid'])).exists()
assert ray['actual_CPU_Ray_copy_and_hook']['old']['HTTP_resume_calls']==0 and ray['actual_CPU_Ray_copy_and_hook']['new']['HTTP_resume_calls']==1
for f in ['CPU_LOCAL_ACCEPTED.json','CPU_NODE_ACCEPTED.json']:assert json.loads((p/f).read_text())['CPU_checks']==24
for f in ['BOUNDARY_CPU_LOCAL_ACCEPTED.json','BOUNDARY_CPU_NODE_ACCEPTED.json']:assert json.loads((p/f).read_text())['checks']==37
assert json.loads((p/'NATIVE_CPU_ACCEPTED.json').read_text())['native_CPU_checks']==4
q=pathlib.Path('/proc/3060432');assert q.exists() and int((q/'stat').read_text().split()[21])==51919204 and q.stat().st_uid==0 and sha(q/'cmdline')=='336761ab09adce6296a5fd5d20491a4c1fc9b37ad46a47f9f3fb3f33ae09013e'
checkpoint=r/'runs/c2-seed1234/checkpoints/latest_checkpointed_iteration.txt';present=checkpoint.exists();value=checkpoint.read_text() if present else None
proof={'accepted':True,'created':time.time(),'source_pins_sha256':sha(p/'SOURCE_PINS.json'),'files_verified':len(s['files']),'candidate_bridge_sha256':sha(p/'bridge.py'),'active_scientific_bridge_unchanged':True,'scientific_owner_live':True,'CPU_cursor_checks_per_environment':24,'CPU_boundary_checks_per_environment':37,'native_CPU_save_load_checks':4,'actual_CPU_Ray_boundary_accepted':True,'CPU_Ray_actor_exactgone':True,'active_admission_gate_present':(r/'management/queue-timeout-recovery-v2/PAUSE_NEW_TASKS.json').exists(),'production_checkpoint_tracker_exists':present,'production_checkpoint_tracker_value':value,'actual_model_optimizer_rng_restored':False,'future_owned_source_binding_pending':True,'new_model_tool_judge_calls':0,'formal_updates':1,'stable':False}
(p/'ROOT_ACCEPTED.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
p=r/'src/slime/slime/utils/misc.py';t=p.read_text();tr=ast.parse(t)
for n in tr.body:
 if isinstance(n,ast.FunctionDef) and n.name=='should_run_periodic_action':print('PERIODIC_SOURCE',ast.get_source_segment(t,n))
