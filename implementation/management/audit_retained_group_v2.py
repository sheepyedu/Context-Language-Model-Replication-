"""Recompute retained first-group costs, token provenance and training masks."""
from argparse import Namespace
from collections import Counter
import hashlib
import json
import math
import argparse
from pathlib import Path

from classify_ancillary_receipt import classify_ancillary_receipt
from polar.rollout.models import TaskResult
from slime_bridge import rollout
from slime_bridge.config import resolve_polar_slime_config
from clm_replication.native_failures import training_view, restore_native_metadata

ROOT=Path('/mount-data/CLM-Replication')
parser=argparse.ArgumentParser()
parser.add_argument('--position',type=int,required=True)
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parameters=parser.parse_args()
path=parameters.input
source_bytes=path.read_bytes()
result=TaskResult.model_validate_json(source_bytes)
assert len(result.results)==32 and len({s.session_id for s in result.results})==32
args=Namespace(**json.loads((ROOT/'runs/c2-seed1234/attempt-05/effective_args.json').read_text()))
config=resolve_polar_slime_config(args)
batch=json.loads((ROOT/'runs/c2-seed1234/audit/rollout-00000/candidate-000/BATCH.json').read_text())
from slime.utils.types import Sample
group=[Sample.from_dict(item) for item in batch['groups'][parameters.position]]
view,dispositions=training_view(result)
samples=rollout._convert_task_result_to_samples(config,view,group,max_tokens=rollout._resolve_max_tokens(args))
restore_native_metadata(samples,dispositions)
actual=Counter(s.metadata['polar']['session_id'] for s in samples)
expected={s.session_id:len(s.trajectory.traces) for s in result.results}
assert dict(actual)==expected
rows=[]
ancillary_dispositions=[]
for session in result.results:
    sid=session.session_id
    archive=ROOT/'runs/session-artifacts'/sid
    manifest=json.loads((archive/'ARCHIVE_MANIFEST.json').read_text())
    for name,expected_hash in manifest.items():
        assert hashlib.sha256((archive/name).read_bytes()).hexdigest()==expected_hash
    events=json.loads((archive/'logs/agent/edit_events.json').read_text())
    traces=session.trajectory.traces
    assert len(events)==len(traces)
    receipts=[json.loads(p.read_text()) for p in (ROOT/'runs/provider-attempts'/sid).glob('*/TOKEN_RECEIPT.json')]
    model_receipts=[]
    for receipt in receipts:
        disposition=classify_ancillary_receipt(ROOT,receipt)
        if disposition is None:
            model_receipts.append(receipt)
        else:
            ancillary_dispositions.append(disposition)
    receipts=model_receipts
    assert len(receipts)==len(traces)
    assert sum(r['input_tokens'] for r in receipts)==sum(len(t.prompt_ids) for t in traces)
    assert sum(r['output_tokens'] for r in receipts)==sum(len(t.response_ids) for t in traces)
    previous=[]
    total=0.0
    for trace in traces:
        assert len(trace.response_ids)==len(trace.response_logprobs)==len(trace.loss_mask)
        assert trace.prompt_messages and trace.response_messages
        reused=0
        for a,b in zip(previous,trace.prompt_ids):
            if a!=b:
                break
            reused+=1
        p,g=len(trace.prompt_ids),len(trace.response_ids)
        total+=13839200000.0*(p-reused+g)+131072.0*((p*p-reused*reused)/2+g*p+g*g/2)
        previous=trace.prompt_ids
    evaluation=session.trajectory.metadata.get('evaluation',{})
    clm=evaluation.get('clm')
    if clm:
        assert math.isclose(total,clm['flops'],rel_tol=1e-12,abs_tol=1)
        assert clm['pure_edit_by_call']==[bool(e['pure_edit']) for e in events]
    mine=[s for s in samples if s.metadata['polar']['session_id']==sid]
    trained=sum(sum(s.loss_mask) for s in mine)
    generated=sum(len(t.response_ids) for t in traces)
    if session.status=='COMPLETED' or sid in dispositions:
        assert trained==generated
    else:
        assert trained==0 and clm is None
    rows.append({'session_id':sid,'native_status':session.status,'model_calls':len(traces),
                 'input_tokens':sum(len(t.prompt_ids) for t in traces),'output_tokens':generated,
                 'reward':evaluation.get('outcome_reward'),'recomputed_flops':total,
                 'archive_files':len(manifest),'pure_edit_turns':sum(bool(e['pure_edit']) for e in events),
                 'trainable_tokens':trained,'training_disposition':dispositions.get(sid)})
assert path.read_bytes()==source_bytes
from clm_replication.bridge import trajectory_outcome
from clm_replication.rewards import group_advantages
whole_group_trainable=False
exclusion='incomplete_or_invalid_reward'
if all(row['reward'] is not None and row['trainable_tokens']>0 for row in rows):
    outcomes={sample.metadata['polar']['session_id']:trajectory_outcome(sample) for sample in samples}
    assert len(outcomes)==32
    group_values=group_advantages(list(outcomes.values()))
    whole_group_trainable=group_values['accepted']
    exclusion=group_values['reason']
proof={'accepted':True,'scope':'complete32 group, native harness/judge, retention and training provenance',
       'source_sha256':hashlib.sha256(source_bytes).hexdigest(),'prompt_count':1,'planned_trajectories':32,
       'native_status_counts':dict(Counter(s.status for s in result.results)),
       'finalized_reward_count':sum(r['reward'] is not None for r in rows),
       'invalid_judge_count':sum(r['reward'] is None for r in rows),
       'derived_closed_tool_timeout_count':len(dispositions),'samples':len(samples),
       'model_calls':sum(r['model_calls'] for r in rows),'pure_edit_turns':sum(r['pure_edit_turns'] for r in rows),
       'all_original_input_ids_logprobs_masks_and_token_receipts_match':True,
       'all_archive_file_hashes_match':True,'native_results_changed':False,
       'whole_group_trainable':whole_group_trainable,'whole_group_exclusion':exclusion,
       'pure_edit_rule':'Pinned official operational criterion: changed mirror, exit0, empty output. Arbitrary silent side effects are not proven absent.',
       'sessions':rows,'ancillary_http_dispositions':ancillary_dispositions,
       'original_unknown_usage_preserved':True,'new_model_tool_or_judge_calls':0}
parameters.output.write_text(json.dumps(proof,indent=2))
print(json.dumps({k:v for k,v in proof.items() if k!='sessions'},indent=2))
