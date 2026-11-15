"""Reload only closed native task records at an idle admission boundary."""
import argparse
import hashlib
import json
import os
import inspect
from pathlib import Path

from polar.rollout.manager import RolloutManager, _TaskRecord
from polar.rollout.models import TaskStatus


class TerminalRecoveredManager(RolloutManager):
    def __init__(self, *, recovery_directory, **kwargs):
        super().__init__(**kwargs)
        directory=Path(recovery_directory)
        plan=json.loads((directory/'PLAN.json').read_bytes())
        self.restored_ids=set()
        for item in plan['tasks']:
            raw=(directory/item['status_file']).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==item['status_sha256']
            status=TaskStatus.model_validate_json(raw)
            assert status.status=='completed'
            assert len(status.results)==status.completed_sessions==status.total_sessions
            metadata=item['metadata']
            assert status.task_id==metadata['task_id']
            assert len({s.session_id for s in status.results})==status.total_sessions
            self._tasks[status.task_id]=_TaskRecord(
                task_id=status.task_id,status=status.status,total_sessions=status.total_sessions,
                completed_sessions=status.completed_sessions,errored_sessions=metadata['errored_sessions'],
                harness=metadata.get('harness'),model=metadata.get('model'),
                created_at=metadata['created_at'],updated_at=metadata['updated_at'],
                results=list(status.results),result_paths=list(status.result_paths),
                session_states={row['session_id']:row['status'] for row in item['sessions']})
            self.restored_ids.add(status.task_id)

    async def submit_task(self, request):
        if request.task_id in self.restored_ids:
            raise ValueError('A restored terminal task cannot be submitted again')
        return await super().submit_task(request)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--topology',required=True)
    parser.add_argument('--recovery-directory',required=True)
    args=parser.parse_args()
    from polar.rollout import pipeline,server
    from pipeline_queue_fixed import Pipeline as FixedPipeline
    pipeline.Pipeline._dispatch_session=FixedPipeline._dispatch_session
    server.RolloutManager=lambda **kwargs:TerminalRecoveredManager(recovery_directory=args.recovery_directory,**kwargs)
    (Path(args.recovery_directory).parent/'RUNTIME_IDENTITY.json').write_text(json.dumps({
        'pid':os.getpid(), 'dispatch_source_sha256':hashlib.sha256(inspect.getsource(pipeline.Pipeline._dispatch_session).encode()).hexdigest(),
        'dispatch_source_file':inspect.getsourcefile(pipeline.Pipeline._dispatch_session),
        'terminal_restore_only':True},indent=2))
    server.serve(args.topology)


if __name__=='__main__':main()
