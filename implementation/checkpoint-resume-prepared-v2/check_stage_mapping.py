"""Test native index/cursor wiring with CPU-only actor handles."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

root=Path('/mount-data/CLM-Replication')
source=root/'src/slime/slime/ray/placement_group.py'
raw=source.read_bytes();tree=ast.parse(raw)
function=next(item for item in tree.body if isinstance(item,ast.FunctionDef) and item.name=='create_training_models')
checks=[]
for loaded in (0,4,9,69):
    args=SimpleNamespace(megatron_config_path=None,use_critic=False,actor_num_nodes=1,
            actor_num_gpus_per_node=4,kl_coef=0,use_kl_loss=True,use_opd=False,
            start_rollout_id=None,rollout_global_dataset=True)
    loaded_cursors=[]
    actor=SimpleNamespace(args=args,async_init=lambda *a,**kw:[loaded+1]*4,
                          set_rollout_manager=lambda manager:None)
    manager=SimpleNamespace(load=SimpleNamespace(remote=lambda index:loaded_cursors.append(index)))
    namespace={'allocate_train_group':lambda **kw:actor,'ray':SimpleNamespace(get=lambda value:value)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(source),'exec'),namespace)
    namespace['create_training_models'](args,{'actor':None},manager)
    assert args.start_rollout_id==loaded+1 and loaded_cursors==[loaded]
    checks.append({'checkpoint_iteration':loaded,'next_policy':args.start_rollout_id,'dataset_cursor_index':loaded_cursors[0]})
proof={'accepted':True,'source_sha256':hashlib.sha256(raw).hexdigest(),'checks':checks,
       'mocked_rank_indices_consistent':True,'new_model_calls':0,'actual_model_or_optimizer_restored':False}
(root/'management/checkpoint-resume-candidate-v2/STAGE_MAPPING_CPU_ACCEPTED.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof,indent=2))
