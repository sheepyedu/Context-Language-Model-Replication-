"""Closed records survive exactly, while old task identities remain single-use."""
import asyncio
from copy import deepcopy
import hashlib,json
from pathlib import Path
import tempfile

from terminal_coordinator import TerminalRecoveredManager
from polar.rollout.models import SessionResult,TaskRequest,TaskStatus

async def main():
 with tempfile.TemporaryDirectory(prefix='terminal-restore-cpu-') as temporary:
    directory=Path(temporary)
    result=SessionResult(session_id='original-terminal',task_id='original-task',status='TIMEOUT',
                         error='original queue timeout',trajectory={'status':'TIMEOUT','traces':[]})
    status=TaskStatus(task_id='original-task',status='completed',total_sessions=1,completed_sessions=1,results=[result])
    raw=status.model_dump_json().encode();(directory/'status.json').write_bytes(raw)
    item={'sessions':[{'session_id':result.session_id,'status':'TIMEOUT'}],'status_file':'status.json','status_sha256':hashlib.sha256(raw).hexdigest(),
          'metadata':{'task_id':status.task_id,'errored_sessions':1,'created_at':10.,'updated_at':20.}}
    plan={'tasks':[item]};(directory/'PLAN.json').write_text(json.dumps(plan))
    manager=TerminalRecoveredManager(recovery_directory=directory,pipeline=None,scheduler=None)
    assert manager.get_task(status.task_id)==status
    request=TaskRequest(task_id=status.task_id,instruction='unused',agent={'import_path':'fixture.module:Harness','model_name':'unused'})
    try:await manager.submit_task(request)
    except ValueError:pass
    else:raise AssertionError('Closed task could be replayed')
    (directory/'status.json').write_text('{}')
    try:TerminalRecoveredManager(recovery_directory=directory,pipeline=None,scheduler=None)
    except AssertionError:pass
    else:raise AssertionError('Changed source was accepted')
 proof={'accepted':True,'cases':3,'original_terminal_result_exact':True,'changed_bytes_rejected':True,
        'old_task_resubmission_rejected':True,'model_calls':0}
 Path(__file__).with_name('TERMINAL_RECOVERY_CPU_ACCEPTED.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
asyncio.run(main())
