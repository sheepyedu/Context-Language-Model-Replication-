"""Cover missing and silently defaulted cursor states without a model."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from checkpoint_cursor import FIELDS, record_loaded_cursor


def setup(root):
    saved = root / 'checkpoint'
    (saved / 'rollout').mkdir(parents=True)
    (saved / 'latest_checkpointed_iteration.txt').write_text('4')
    state = {'sample_offset': 80, 'epoch_id': 0, 'sample_group_index': 80,
             'sample_index': 2560, 'metadata': {'fixture': True}}
    path = saved / 'rollout/global_dataset_state_dict_4.pt'
    path.write_text(json.dumps(state))
    args = SimpleNamespace(load=str(saved), start_rollout_id=5, ckpt_step=None,
                           finetune=False, no_load_optim=False, no_load_rng=False,
                           clm_audit_dir=str(root / 'audit'))
    return args, state, path


checks = []
load_json = lambda path: json.loads(path.read_text())
for case in ('round_trip', 'repeat_same_proof', 'explicit_older_iteration', 'serialized_start_is_none'):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        args, state, path = setup(root)
        if case == 'serialized_start_is_none':
            args.start_rollout_id = None
        if case == 'explicit_older_iteration':
            Path(args.load, 'latest_checkpointed_iteration.txt').write_text('9')
            args.ckpt_step = 4
        proof = record_loaded_cursor(args, 5, SimpleNamespace(**state), load_json)
        assert proof['accepted'] and not proof['actual_model_optimizer_rng_restore_asserted']
        if case == 'repeat_same_proof':
            assert proof == record_loaded_cursor(args, 5, SimpleNamespace(**state), load_json)
        checks.append(case)

invalid = ('missing_load', 'wrong_explicit_iteration', 'missing_file', 'release_tracker', 'wrong_iteration', 'wrong_start',
           'finetune', 'no_load_optim', 'no_load_rng', 'invalid_counter',
           *['missing_' + field for field in FIELDS],
           *['loaded_wrong_' + field for field in FIELDS])
for case in invalid:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        args, state, path = setup(root)
        source = SimpleNamespace(**state)
        if case == 'missing_load':
            args.load = None
        elif case == 'wrong_explicit_iteration':
            args.ckpt_step = 3
        elif case == 'missing_file':
            path.unlink()
        elif case == 'release_tracker':
            Path(args.load, 'latest_checkpointed_iteration.txt').write_text('release')
        elif case == 'wrong_iteration':
            Path(args.load, 'latest_checkpointed_iteration.txt').write_text('3')
        elif case == 'wrong_start':
            args.start_rollout_id = 4
        elif case in ('finetune', 'no_load_optim', 'no_load_rng'):
            setattr(args, case, True)
        elif case == 'invalid_counter':
            state['sample_index'] = -1
            path.write_text(json.dumps(state))
        elif case.startswith('missing_'):
            del state[case.removeprefix('missing_')]
            path.write_text(json.dumps(state))
        elif case.startswith('loaded_wrong_'):
            field = case.removeprefix('loaded_wrong_')
            setattr(source, field, {} if field == 'metadata' else 0)
            if field == 'epoch_id':
                source.epoch_id = 1
        try:
            record_loaded_cursor(args, 5, source, load_json)
        except (ValueError, FileNotFoundError):
            checks.append(case)
            continue
        raise AssertionError('Invalid resume cursor accepted for ' + case)

print(json.dumps({'accepted': True, 'CPU_checks': len(checks), 'checks': checks,
                  'model_tool_judge_calls': 0, 'actual_model_restore': False}))
