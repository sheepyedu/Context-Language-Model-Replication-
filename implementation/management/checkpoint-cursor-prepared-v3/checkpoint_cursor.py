"""Verify the actual loaded dataset cursor before resumed task admission."""

import hashlib
import json
import os
from pathlib import Path


FIELDS = ('sample_offset', 'epoch_id', 'sample_group_index', 'sample_index', 'metadata')


def record_loaded_cursor(args, policy, data_source, loader=None):
    if not isinstance(policy, int) or isinstance(policy, bool) or policy <= 0:
        raise ValueError('This cursor proof applies to a nonzero resumed policy')
    if args.start_rollout_id is not None and args.start_rollout_id != policy:
        raise ValueError('Resumed policy differs from the native start index')
    if any(bool(getattr(args, field, False)) for field in
           ('finetune', 'no_load_optim', 'no_load_rng')):
        raise ValueError('Resume configuration skips the saved training state')
    if not args.load:
        raise ValueError('Resume checkpoint location is missing')
    root = Path(args.load)
    expected_iteration = policy - 1
    selected_iteration = getattr(args, 'ckpt_step', None)
    if selected_iteration is None:
        tracker = (root / 'latest_checkpointed_iteration.txt').read_text().strip()
        if not tracker.isdigit():
            raise ValueError('Training resume requires a numeric saved iteration')
        selected_iteration = int(tracker)
    if selected_iteration != expected_iteration:
        raise ValueError('Selected checkpoint does not precede the resumed policy')
    path = root / 'rollout' / f'global_dataset_state_dict_{expected_iteration}.pt'
    if loader is None:
        import torch

        loader = lambda value: torch.load(value, map_location='cpu', weights_only=False)
    raw = path.read_bytes()
    expected = loader(path)
    if not isinstance(expected, dict) or any(field not in expected for field in FIELDS):
        raise ValueError('Saved cursor is incomplete')
    for field in FIELDS[:-1]:
        value = expected[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError('Invalid saved cursor counter ' + field)
    actual = {field: getattr(data_source, field) for field in FIELDS}
    expected = {field: expected[field] for field in FIELDS}
    if expected != actual:
        raise ValueError('Native in-memory cursor differs from the saved cursor')
    if raw != path.read_bytes():
        raise ValueError('Saved cursor changed during verification')
    proof = {'accepted': True, 'scope': 'saved and actually loaded dataset cursor only',
             'policy': policy, 'serialized_start_rollout_id': args.start_rollout_id,
             'checkpoint_iteration': expected_iteration,
             'source': str(path), 'source_sha256': hashlib.sha256(raw).hexdigest(),
             'expected': expected, 'actual': actual, 'pid': os.getpid(),
             'actual_model_optimizer_rng_restore_asserted': False}
    destination = Path(args.clm_audit_dir) / f'loaded-cursor-policy-{policy:05d}-{os.getpid()}.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if json.loads(destination.read_text()) != proof:
            raise ValueError('Previously recorded cursor proof differs')
    else:
        with destination.open('x') as handle:
            json.dump(proof, handle, indent=2)
    return proof
