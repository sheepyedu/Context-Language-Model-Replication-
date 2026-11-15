"""Check the lifecycle-only admission gate and the exact original task payloads."""
from argparse import Namespace
import ast,asyncio,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import tempfile

from clm_replication.bridge import wait_for_task_admission

root=Path('/mount-data/CLM-Replication');directory=root/'management/queue-timeout-recovery-v2'
async def gates():
    with tempfile.TemporaryDirectory(prefix='clm-admission-cpu-') as temporary:
        path=Path(temporary);gate=path/'pause.json';gate.write_text('{}')
        args=SimpleNamespace(clm_admission_gate_path=str(gate),clm_audit_dir=str(path))
        waiting=asyncio.create_task(wait_for_task_admission(args,0,0,10))
        await asyncio.sleep(.03)
        assert not waiting.done()
        assert (path/'control-admission-00000-000-010-paused.json').exists()
        gate.unlink();await waiting
        assert (path/'control-admission-00000-000-010-resumed.json').exists()
        await wait_for_task_admission(SimpleNamespace(),0,0,1)
asyncio.run(gates())
old=ast.parse((root/'source-revisions/training-v6/clm_replication/bridge.py').read_text())
new=ast.parse((root/'source-revisions/training-v7/clm_replication/bridge.py').read_text())
for name in ('candidate_decision','should_admit_candidate','post_process_rewards','trajectory_outcome'):
    a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
    b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(a)==ast.dump(b),name
# Only the unstarted branch reaches the gate, and each group opens a fresh transport after it.
collector=next(n for n in new.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='collect_candidate_groups')
loop=next(n for n in collector.body if isinstance(n,ast.For))
condition=next(n for n in loop.body if isinstance(n,ast.If) and ast.unparse(n.test)=='not started')
assert 'await wait_for_task_admission' in ast.unparse(condition)
client=next(n for n in loop.body if isinstance(n,ast.AsyncWith))
assert client.lineno>condition.lineno
from slime_bridge import rollout
from slime_bridge.config import resolve_polar_slime_config
from slime.utils.types import Sample
args=Namespace(**json.loads((directory/'validation/effective_args.json').read_bytes()))
assert args.clm_source_revision=='training-v7' and args.num_rollout==70 and args.rollout_batch_size==8
assert args.n_samples_per_prompt==32 and args.num_steps_per_rollout==1
config=resolve_polar_slime_config(args)
candidate=root/'runs/c2-seed1234/audit/rollout-00000/candidate-000'
batch=json.loads((candidate/'BATCH.json').read_bytes());count=0
for intent in candidate.glob('group-*-intent.json'):
    position=int(intent.name.split('-')[1]);group=[Sample.from_dict(item) for item in batch['groups'][position]]
    payload=rollout._build_task_payload(args=args,config=config,group=group,rollout_id=0,task_position=position)
    payload['task_id']=f'clm-{args.clm_run_id}-00000-000-{position:03d}'
    rollout._attach_scheduler_metadata(payload,group_id=group[0].group_index,policy_version=0,rollout_step=0)
    assert payload==json.loads(intent.read_bytes());count+=1
proof={'accepted':True,'gate_blocks_only_new_task_creation':True,'existing_intents_bypass_gate':True,
       'unchanged_scientific_functions_ast':4,'exact_original_task_payloads':count,
       'fresh_group_transport_after_gate':True,'logical_batch':256,'target_updates':70,'model_calls':0}
(directory/'ADMISSION_CPU_ACCEPTED.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
