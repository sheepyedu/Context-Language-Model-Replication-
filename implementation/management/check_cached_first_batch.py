"""Rebuild the frozen first batch offline and compare every native Sample."""

from argparse import Namespace
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
from slime.rollout.data_source import RolloutDataSource
from clm_replication import bridge

root = Path('/mount-data/CLM-Replication')
args = Namespace(**json.loads((root / 'runs/c2-seed1234/attempt-07/effective_args.json').read_bytes()))
os.environ.pop('CLM_RESUME_INITIAL_GATEWAY', None)
bridge._served_policy_version = 0
source = RolloutDataSource(args)
network_calls = []


def reject_network(*a, **kw):
    network_calls.append((a, kw))
    raise AssertionError('A cached batch attempted a new network request')


with patch('httpx.AsyncClient.post', side_effect=reject_network), \
     patch('httpx.AsyncClient.get', side_effect=reject_network):
    output = bridge.generate_rollout(args, 0, source)
samples = [sample.to_dict() for group in output.samples for sample in group]
original = torch.load(root / 'runs/c2-seed1234/training/rollout-0.pt', weights_only=False)['samples']
assert len(samples) == len(original) == 11095


def digest(sample):
    return hashlib.sha256(json.dumps(sample, sort_keys=True, default=str).encode()).hexdigest()


assert all(digest(a) == digest(b) for a, b in zip(samples, original))
assert not network_calls
proof = {'accepted': True, 'samples': len(samples), 'logical_trajectories': 256,
         'prompt_groups': 8, 'all_native_sample_fields_identical': True,
         'new_model_tool_judge_or_network_calls': 0,
         'selected_groups': sorted({s['group_index'] for s in samples}),
         'cursor_after': {key: getattr(source, key) for key in
                          ('epoch_id', 'sample_group_index', 'sample_index', 'sample_offset', 'metadata')}}
(root / 'management/CACHED_FIRST_BATCH_ROOT_ACCEPTED.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof, indent=2))
