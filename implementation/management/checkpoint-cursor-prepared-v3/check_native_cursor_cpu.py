"""Exercise the frozen native save/load methods on CPU and check the guard."""

import ast
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace

import torch

from checkpoint_cursor import FIELDS, record_loaded_cursor


path = Path('/mount-data/CLM-Replication/src/slime/slime/rollout/data_source.py')
raw = path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '446595578aff88fac1637b8956a245853db1167e7964896bbab2fff61b1db616'
tree = ast.parse(raw)
methods = [next(node for node in reversed(list(ast.walk(tree)))
                if isinstance(node, ast.FunctionDef) and node.name == name)
           for name in ('save', 'load')]
namespace = {'torch': torch, 'os': os, 'logger': logging.getLogger('native-cursor-fixture')}
exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(path), 'exec'), namespace)
checks = []
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    checkpoint = root / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'latest_checkpointed_iteration.txt').write_text('4')
    args = SimpleNamespace(save=str(checkpoint), load=str(checkpoint), rollout_global_dataset=True,
                           rollout_shuffle=True, start_rollout_id=None, ckpt_step=None,
                           no_load_optim=False, no_load_rng=False, finetune=False,
                           clm_audit_dir=str(root / 'audit'))
    expected = {'sample_offset': 80, 'epoch_id': 2, 'sample_group_index': 80,
                'sample_index': 2560, 'metadata': {'native_fixture': True}}
    shuffled = []
    source = SimpleNamespace(args=args, dataset=SimpleNamespace(shuffle=shuffled.append), **expected)
    source.save = MethodType(namespace['save'], source)
    source.load = MethodType(namespace['load'], source)
    source.save(4)
    for field in FIELDS:
        setattr(source, field, {} if field == 'metadata' else 0)
    source.load(4)
    assert {field: getattr(source, field) for field in FIELDS} == expected
    assert record_loaded_cursor(args, 5, source)['accepted']
    checks.append('native_full_cursor_round_trip_and_guard')
    assert shuffled == [2]
    checks.append('native_epoch_shuffle_restored')
    cursor = checkpoint / 'rollout/global_dataset_state_dict_4.pt'
    cursor.unlink()
    for field in FIELDS:
        setattr(source, field, {} if field == 'metadata' else 0)
    source.load(4)
    assert source.sample_index == 0
    try:
        record_loaded_cursor(args, 5, source)
    except FileNotFoundError:
        checks.append('native_missing_file_silent_reset_detected')
    else:
        raise AssertionError('Missing cursor was accepted')
    torch.save({'sample_offset': 80}, cursor)
    source.load(4)
    assert source.sample_index == 0
    try:
        record_loaded_cursor(args, 5, source)
    except ValueError:
        checks.append('native_missing_fields_silent_defaults_detected')
    else:
        raise AssertionError('Partial cursor was accepted')
assert not torch.cuda.is_initialized()
proof = {'accepted': True, 'native_CPU_checks': len(checks), 'checks': checks,
         'native_source_sha256': hashlib.sha256(raw).hexdigest(),
         'CUDA_initialized': False, 'actual_model_optimizer_rng_restored': False,
         'production_checkpoint_read_or_modified': False}
Path(__file__).with_name('NATIVE_CPU_ACCEPTED.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof))
