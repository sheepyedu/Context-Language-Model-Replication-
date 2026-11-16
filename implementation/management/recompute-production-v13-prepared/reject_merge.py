"""Independently retain the failed numerical decision and original tolerances."""
import hashlib,json,time
from pathlib import Path
R=Path('/mount-data/CLM-Replication');P=R/'management/strict-merge-native-prepared-v6'
result=json.loads((P/'RESULT.json').read_text());assert not result['accepted']
assert result['child_integer_exit']==1 and not result['remaining_matching'] and not result['timed_out']
pid=result['owner_pid'];assert not Path(f'/proc/{pid}').exists()
proof={'rejected':True,'created':time.time(),'all_owned_gone':True,'owner_pid':pid,'original_child_integer_exit':1,'actual_optimizer_updates':0,'new_generation_calls':0,'full_suite_completed':False,'rejection_scope':'first retained short two-call case exceeded fixed per-parameter gradient tolerance; later cases not evaluated','rank_evidence':[],'production_merge_active':False,'proposed_followup':'continue unmerged retained batch, validate recomputation-group-only memory optimization before reference/PPO'}
for rank in range(4):
 p=P/f'native-rank-{rank}/case-0.json';d=json.loads(p.read_text());rr=result['rank_results'][str(rank)]
 assert not rr['accepted'] and rr['parameter_sha256_before']==rr['parameter_sha256_after']
 assert rr['RNG_restored'] and rr['gradients_cleared'] and rr['actual_optimizer_updates']==0
 g=d['gradient_comparison'];bad=[x for x in g['tensors'] if not x['accepted']]
 assert not g['accepted'] and bad
 proof['rank_evidence'].append({'rank':rank,'case_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'logprob_max_abs':d['logprob_max_abs'],'loss_abs_diff':d['loss_abs_diff'],'gradient_relative_l2':g['relative_l2'],'failing_tensors':len(bad),'total_tensors':len(g['tensors']),'worst_ratio':max(x['max_abs_diff']/x['tolerance'] for x in bad),'warm_subset_speedup':rr['warm_compute_speedup']})
(R/'management/STRICT_MERGE_NATIVE_V6_ROOT_REJECTED.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
