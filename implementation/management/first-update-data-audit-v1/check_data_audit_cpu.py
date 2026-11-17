"""Reject mismatched masks, identities, rewards and token credits."""

import copy
import json
from pathlib import Path

import torch

from audit_first_update_data import check_row


sample = {'response_length': 2, 'index': 7, 'tokens': [3, 4, 5],
          'loss_mask': [1, 1], 'rollout_log_probs': [-0.2, -0.3]}
expected = {'raw': 1.0, 'task': 0.2, 'efficiency': 0.4, 'role': 1}
row = {'response_length': 2, 'sample_index': 7, 'tokens': torch.tensor([3, 4, 5]),
       'total_length': 3, 'loss_mask': torch.tensor([1, 1]),
       'rollout_log_probs': torch.tensor([-0.2, -0.3]), 'role_mask': torch.ones(2),
       'efficiency': 0.4, 'reward': 0.2, 'raw_reward': 1.0,
       'ref_log_probs': torch.tensor([-0.1, -0.1]), 'kl': torch.zeros(2)}
credit = torch.full((2,), 0.2) + 0.25 * 0.4 * torch.ones(2)
row['advantages'] = credit.clone()
row['returns'] = credit.clone()
check_row(row, sample, expected)
bad_values = {
    'response_length': 1, 'sample_index': 8, 'tokens': torch.tensor([3, 4, 6]),
    'total_length': 4, 'loss_mask': torch.tensor([1, 0]),
    'rollout_log_probs': torch.tensor([-0.2, -0.4]), 'role_mask': torch.tensor([1.0, 0.0]),
    'efficiency': 0.5, 'reward': 0.3, 'raw_reward': 0.0,
    'ref_log_probs': torch.tensor([float('nan'), 0]), 'kl': torch.zeros(3),
    'advantages': torch.full((2,), 0.2), 'returns': torch.full((2,), 0.2),
}
for field, value in bad_values.items():
    candidate = copy.deepcopy(row)
    candidate[field] = value
    try:
        check_row(candidate, sample, expected)
    except AssertionError:
        continue
    raise AssertionError('Invalid data accepted for ' + field)
result = {'accepted': True, 'checks': 1 + len(bad_values),
          'invalid_fields': list(bad_values), 'cuda_initialized': torch.cuda.is_initialized(),
          'scope': 'CPU fixtures for single-row acceptance; actual native data is separate'}
Path(__file__).with_name('CPU_NODE_ACCEPTED.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
