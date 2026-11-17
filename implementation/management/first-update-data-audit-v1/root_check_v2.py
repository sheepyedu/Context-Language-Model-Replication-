import pathlib,json,hashlib,time,signal
signal.alarm(35)
r=pathlib.Path('/mount-data/CLM-Replication');p=r/'management/first-update-data-audit-v1'
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
pins=json.loads((p/'SOURCE_PINS_v2.json').read_text())
for f,h in pins['files'].items():assert sha(p/f)==h,f
a=json.loads((p/'ACCEPTED.json').read_text());assert a['accepted'] and a['counts']['samples']==11095 and a['logical_trajectories']==256 and len(a['groups'])==8
assert json.loads((p/'OWNER_EXIT.json').read_text())['exit_code']==0
assert json.loads((p/'EXIT.json').read_text())['exit_code']==0
for f in ['OWNER_LAUNCH.json','WORKER_LAUNCH.json']:assert not pathlib.Path('/proc',str(json.loads((p/f).read_text())['pid'])).exists()
summary=r/'runs/c2-seed1234/audit/sample-summary-policy-00000.json'
s=json.loads(summary.read_text());c=a['counts']
assert s['trajectory_count']==256 and s['prompt_group_count']==8 and s['model_call_samples']==11095
assert s['input_token_occurrences']==c['input_token_occurrences'] and s['output_token_occurrences']==c['output_tokens']
assert s['efficiency_role_tokens']==c['edit_role_tokens'] and s['trainable_generated_tokens']==c['loss_tokens']
q=pathlib.Path('/proc/3060432');assert q.exists() and int((q/'stat').read_text().split()[21])==51919204 and q.stat().st_uid==0
assert sha(q/'cmdline')==a['owner']['raw_sha256']
proof={'accepted':True,'created':time.time(),'source_pins_sha256':sha(p/'SOURCE_PINS_v2.json'),'actual_acceptance_sha256':sha(p/'ACCEPTED.json'),'native_sample_summary_sha256':sha(summary),'samples':11095,'logical_trajectories':256,'saved_rank':a['saved_rank'],'all_rank_buffers_separately_retained':False,'successful_trajectories':sum(g['successes'] for g in a['groups']),'zero_success_groups':sum(g['successes']==0 for g in a['groups']),'actual_wait':0,'owner_and_worker_exactgone':True,'scientific_owner_identity_matches':True,'formal_updates':1,'three_update_stability_accepted':False,'checkpoint_restore_accepted':False,'new_model_tool_judge_calls':0}
(p/'ROOT_ACCEPTED.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
