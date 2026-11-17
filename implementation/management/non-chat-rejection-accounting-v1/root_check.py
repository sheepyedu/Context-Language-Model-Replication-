import pathlib,json,hashlib,time
r=pathlib.Path('/mount-data/CLM-Replication');p=r/'management/non-chat-rejection-accounting-v1';sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
s=json.loads((p/'SOURCE_PINS.json').read_text())
for name,h in s['files'].items():assert sha(p/name)==h,name
assert sha(r/'management/token_report_v4.py')==s['base_token_report_v4_sha256']
a=json.loads((p/'ACTUAL_REJECTIONS.json').read_text());assert a['count']==len(a['records'])==8
for v in a['records']:
 ledger=json.loads((r/'runs/token-ledger'/str(v['call_id']+'.json')).read_text());d=pathlib.Path(ledger['source_directory'])
 for name,h in v['source_sha256'].items():assert sha(d/name)==h
 assert json.loads((d/'TOKEN_RECEIPT.json').read_text())==ledger
 assert all(ledger.get(k) is None for k in ['input_tokens','output_tokens','total_tokens','model_reported','response_id'])
 req=json.loads((d/'INTENT.json').read_text())['request'];err=json.loads((d/'ERROR.json').read_text())
 assert 'messages' not in req and req['action'] in ['search','fetch'] and err['body']['error']['code']=='json_parse_error'
t=json.loads((p/'ACTUAL_TOKEN_REPORT.json').read_text());k=t['by_request_kind'];assert k['non_chat_payload_schema_rejection']['provider_attempts']==7 and k['accepted_ancillary_http_schema_rejection']['provider_attempts']==1
assert k['provider_model_request']['missing_total_tokens_calls']==0 and t['all']['missing_total_tokens_calls']==8
assert t['all_dimensions_reconcile'] and t['csv_all_roles_reconcile']
for f in ['known_input_tokens','known_output_tokens','known_total_tokens','provider_attempts']:assert sum(v[f] for v in k.values())==t['all'][f]
q=pathlib.Path('/proc/3060432');assert q.exists() and int((q/'stat').read_text().split()[21])==51919204 and sha(q/'cmdline')=='336761ab09adce6296a5fd5d20491a4c1fc9b37ad46a47f9f3fb3f33ae09013e'
proof={'accepted':True,'created':time.time(),'source_pins_sha256':sha(p/'SOURCE_PINS.json'),'private_rejection_evidence_sha256':sha(p/'ACTUAL_REJECTIONS.json'),'token_report_sha256':sha(p/'ACTUAL_TOKEN_REPORT.json'),'canonical_receipts':t['all']['provider_attempts'],'model_and_engineering_receipts':k['provider_model_request']['provider_attempts'],'model_receipt_missing_usage':0,'original_ancillary_HTTP_requests':1,'new_non_chat_schema_rejections':7,'all_original_missing_usage_preserved':8,'known_total_tokens':t['all']['known_total_tokens'],'judge_known_tokens':t['by_role']['judge']['known_total_tokens'],'money':None,'caller_command_provenance':'pending original session artifact completion','original_v4_unchanged':True,'scientific_owner_live':True,'new_model_tool_judge_calls':0,'formal_updates':1,'checkpoint_restore_accepted':False,'stable':False}
(p/'ROOT_ACCEPTED.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
