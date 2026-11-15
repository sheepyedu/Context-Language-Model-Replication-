"""Export the quiescent coordinator before replacement, preserving original IDs."""
from pathlib import Path
import hashlib
import json
import os
import time
import urllib.request

ROOT=Path('/mount-data/CLM-Replication')
OUTPUT=ROOT/'management/queue-timeout-recovery-v2/snapshot'

def get(path):
    with urllib.request.urlopen('http://127.0.0.1:19170'+path,timeout=120) as response:
        return response.read()

def main():
    assert not OUTPUT.exists(), 'Existing snapshot requires reconciliation, not replacement'
    OUTPUT.mkdir()
    candidates={}
    for path in (ROOT/'runs/c2-seed1234/audit').glob('rollout-*/candidate-*/group-*-intent.json'):
        value=json.loads(path.read_bytes());candidates[value['task_id']]=path
    metadata=json.loads(get('/tasks?limit=1000'))['tasks']
    assert all(row['status']=='completed' for row in metadata)
    known={row['task_id'] for row in metadata}
    assert set(candidates)<=known, 'A submitted intent lacks a coordinator record; inspect actual effect'
    tasks=[];pending_count=0
    for index,meta in enumerate(metadata):
        task_id=meta['task_id']
        raw=get('/rollout/task/'+task_id);status=json.loads(raw)
        sessions=json.loads(get('/tasks/'+task_id+'/sessions'))['sessions']
        ids=[row['session_id'] for row in sessions]
        completed={row['session_id'] for row in status['results']}
        pending=[sid for sid in ids if sid not in completed]
        assert len(set(ids))==len(ids)==status['total_sessions']
        assert status['completed_sessions']==len(completed)==len(status['results'])
        assert len(pending)+len(completed)==status['total_sessions']
        item={'metadata':meta,'sessions':sessions,'status_file':f'task-{index:03d}.json',
              'status_sha256':hashlib.sha256(raw).hexdigest(),
              'original_session_order':ids,'pending_session_ids':pending}
        if pending:
            assert status['status']=='running' and task_id in candidates
            assert all(row['status']=='DISPATCHING' for row in sessions if row['session_id'] in pending)
            intent_count=sum(len(list((ROOT/'runs/provider-attempts'/sid).glob('*/INTENT.json'))) for sid in pending)
            archive_count=sum((ROOT/'runs/session-artifacts'/sid).exists() for sid in pending)
            assert intent_count==archive_count==0
            # A newly submitted cohort has ample separation from its old dispatch deadline.
            assert time.time()-meta['created_at'] < 7200
            request_raw=candidates[task_id].read_bytes()
            item.update(request_file=f'request-{index:03d}.json',request_sha256=hashlib.sha256(request_raw).hexdigest(),
                        pending_original_model_request_count=intent_count,pending_original_archive_count=archive_count,
                        original_queue_wait_exact_seconds=None)
            (OUTPUT/item['request_file']).write_bytes(request_raw)
            pending_count+=len(pending)
        else:
            assert status['status'] in ('completed','failed')
        (OUTPUT/item['status_file']).write_bytes(raw)
        tasks.append(item)
    # No new submissions or callbacks arrived while exporting the quiescent state.
    after=json.loads(get('/tasks?limit=1000'))['tasks']
    before_identity={x['task_id']:(x['status'],x['completed_sessions'],x['updated_at']) for x in metadata}
    after_identity={x['task_id']:(x['status'],x['completed_sessions'],x['updated_at']) for x in after}
    assert before_identity==after_identity
    proof={'created':time.time(),'pending_policy_version':0,'tasks':tasks,
           'restored_terminal_tasks':sum(not item['pending_session_ids'] for item in tasks),
           'original_pending_sessions':pending_count,'terminal_results_changed':False,
           'old_coordinator_pid':2345487,'new_model_tool_or_judge_calls':0}
    (OUTPUT/'PLAN.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps({key:value for key,value in proof.items() if key!='tasks'},indent=2))

if __name__=='__main__':main()
