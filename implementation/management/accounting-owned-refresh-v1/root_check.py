import pathlib,json,hashlib,time
r=pathlib.Path('/mount-data/CLM-Replication');p=r/'management/accounting-owned-refresh-v1';d=p/'refresh-00001';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
i=json.loads((d/'INTENT.json').read_text());e=json.loads((d/'EXIT.json').read_text());s=json.loads((d/'SNAPSHOT.json').read_text());c=json.loads((d/'CHILD_LAUNCH.json').read_text())
current=json.loads((p/'CURRENT.json').read_text())
assert sha(p/'run_refresh.py')==current['source_sha256']
assert current['pid']==i['owner']['pid'] and current['start_ticks']==i['owner']['start_ticks']
pins=json.loads((p/'SOURCE_PINS.json').read_text())
for name,h in pins['files'].items():
 source=p/name if (p/name).exists() else d/name
 assert sha(source)==h,name
assert sha(pathlib.Path(i['script']))==i['script_sha256']
assert e['child_actual_exit_code']==0 and e['publication_exit_code']==0
assert not pathlib.Path('/proc',str(i['owner']['pid'])).exists() and not pathlib.Path('/proc',str(c['pid'])).exists()
assert s['all_dimensions_reconcile'] and s['csv_all_roles_reconcile']
k=s['by_request_kind'];assert sum(v['provider_attempts'] for v in k.values())==s['all']['provider_attempts'] and sum(v['known_total_tokens'] for v in k.values())==s['all']['known_total_tokens']
q=pathlib.Path('/proc/3060432');assert q.exists() and int((q/'stat').read_text().split()[21])==51919204 and sha(q/'cmdline')=='336761ab09adce6296a5fd5d20491a4c1fc9b37ad46a47f9f3fb3f33ae09013e'
proof={'accepted':True,'created':time.time(),'source_sha256':sha(p/'run_refresh.py'),'source_pins_sha256':sha(p/'SOURCE_PINS.json'),'child_actual_exit_code':0,'publication_exit_code':0,'owner_and_child_exactgone':True,'actual_duration_seconds':e['created']-i['created'],'snapshot_sha256':sha(d/'SNAPSHOT.json'),'canonical_receipts':s['all']['provider_attempts'],'known_total_tokens':s['all']['known_total_tokens'],'judge_known_tokens':s['by_role']['judge']['known_total_tokens'],'original_missing_usage':s['all']['missing_total_tokens_calls'],'model_receipt_missing_usage':k['provider_model_request']['missing_total_tokens_calls'],'scientific_owner_live':True,'formal_updates':1,'all_failure_branches_tested':False,'new_model_tool_judge_calls':0}
(p/'ROOT_ACCEPTED.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
