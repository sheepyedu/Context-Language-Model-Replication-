import pathlib,json,hashlib,subprocess,time
r=pathlib.Path('/mount-data/CLM-Replication');p=r/'management/train-dump-namespace-prepared-v1';source=r/'management/train-v14.sh';output=p/'train-v14-rank-debug.sh'
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
original_sha='3e289d4b40700f8d191c80fe5d67afc597997032d8e42c33112283722b81d366'
assert sha(source)==original_sha
command=['python3',str(p/'prepare_train_dump_namespace.py'),'--input',str(source),'--output',str(output),'--expected-input-sha256',original_sha]
z=subprocess.run(command,capture_output=True,text=True,timeout=10)
assert z.returncode==0,z.stderr
receipt=json.loads(z.stdout)
assert receipt['new_template']=='$RUN/training-attempt13/train-{rollout_id}-rank-{rank}.pt'
assert source.read_text().replace(receipt['old_template'],receipt['new_template'],1)==output.read_text()
assert sha(source)==original_sha
syntax=subprocess.run(['bash','-n',str(output)],capture_output=True,text=True,timeout=10);assert syntax.returncode==0,syntax.stderr
failed=[]
for name,cmd in [('overwrite_prepared',command),('overwrite_original',[*command[:5],str(source),*command[6:]]),('wrong_input_sha',[*command[:-1],'0'*64])]:
 n=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
 assert n.returncode==1,name
 failed.append({'case':name,'exit_code':n.returncode,'stderr':n.stderr[-600:]})
assert sha(source)==original_sha and sha(output)==receipt['output_sha256']
owner=pathlib.Path('/proc/3060432');assert int((owner/'stat').read_text().split()[21])==51919204 and owner.stat().st_uid==0
assert sha(owner/'cmdline')=='336761ab09adce6296a5fd5d20491a4c1fc9b37ad46a47f9f3fb3f33ae09013e'
proof={'accepted':True,'created':time.time(),'actual_prepare_CLI_exit':0,'bash_syntax_exit':syntax.returncode,'negative_CLI_cases':failed,'receipt':receipt,'original_live_launcher_unchanged':True,'same_scientific_owner_live':True,'active_training_modified':False,'future_entry_binding_pending':True,'actual_training_launches':0,'GPU_model_judge_calls':0}
(p/'ROOT_ACCEPTED.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
