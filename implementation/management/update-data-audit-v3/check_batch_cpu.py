"""Exercise the generic audit with a complete CPU-only 8x32 synthetic batch."""

import copy
import json
from pathlib import Path
import statistics
import tempfile
from unittest import mock

import torch

import audit_update_data as audit


def fixture(policy):
    samples = []
    rows = []
    for group in range(16, 24):
        outcomes = []
        for position in range(32):
            outcomes.append({'success': position < 2, 'clean_finish': True,
                             'no_tool': False,
                             'format_failure': position >= 2 and position % 5 == 0,
                             'flops': 100.0 + position,
                             'pure_edit_by_call': [position % 2 == 0]})
        rewards = [1.0 if item['success'] else -0.2 * item['format_failure']
                   for item in outcomes]
        mean = statistics.mean(rewards)
        std = statistics.stdev(rewards)
        successful_mean = statistics.mean(item['flops'] for item in outcomes if item['success'])
        for position, outcome in enumerate(outcomes):
            index = len(samples)
            sample = {'response_length': 1, 'index': index, 'tokens': [3, 4],
                      'loss_mask': [1], 'rollout_log_probs': [-0.2],
                      'group_index': group,
                      'metadata': {'polar': {'session_id': f'fixture-{group}-{position}',
                                             'trace_index': 0,
                                             'trajectory_metadata': {
                                                 'policy_version': policy,
                                                 'evaluation': {'clm': outcome}}}}}
            task = (rewards[position] - mean) / (std + 1e-6)
            efficiency = ((successful_mean - outcome['flops']) / successful_mean
                          if outcome['success'] else 0.0)
            role = torch.tensor([int(outcome['pure_edit_by_call'][0])], dtype=torch.float32)
            credit = torch.full((1,), task) + 0.25 * efficiency * role
            row = {'tokens': torch.tensor(sample['tokens'], dtype=torch.int64),
                   'response_lengths': 1, 'sample_indices': index, 'total_lengths': 2,
                   'loss_masks': torch.tensor([1]),
                   'rollout_log_probs': torch.tensor([-0.2]), 'role_masks': role,
                   'a_eff': efficiency, 'rewards': task, 'raw_reward': rewards[position],
                   'ref_log_probs': torch.tensor([-0.3]), 'kl': torch.zeros(1),
                   'advantages': credit, 'returns': credit.clone()}
            samples.append(sample)
            rows.append(row)
    data = {key: [row[key] for row in rows] for key in rows[0]}
    data['dynamic_global_batch_size'] = len(samples)
    return {'rollout_id': policy, 'samples': samples}, {
        'rollout_id': policy, 'rank': 1, 'rollout_data': data}


def main():
    owner = {'pid': 1, 'uid': 0, 'start_ticks': 123, 'raw_sha256': 'fixture'}
    passed = 0
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        rollout, saved = fixture(1)

        def run(left, right, policy=1):
            torch.save(left, directory / f'rollout-{policy}.pt')
            torch.save(right, directory / f'train-{policy}.pt')
            with mock.patch.object(audit, 'native_identity', return_value=owner):
                return audit.audit(directory, policy, owner)

        accepted = run(rollout, saved)
        assert accepted['accepted'] and accepted['logical_trajectories'] == 256
        assert accepted['counts']['samples'] == 256
        assert accepted['prospective_formal_update'] == 2
        assert not accepted['parameter_update_accepted']
        assert not accepted['cuda_initialized']
        passed += 1
        for kind in ['policy', 'sample_count', 'missing_trajectory', 'nonfinite_flops',
                     'credit', 'wrong_saved_policy']:
            left, right = copy.deepcopy(rollout), copy.deepcopy(saved)
            if kind == 'policy':
                left['samples'][0]['metadata']['polar']['trajectory_metadata']['policy_version'] = 0
            elif kind == 'sample_count':
                right['rollout_data']['dynamic_global_batch_size'] += 1
            elif kind == 'missing_trajectory':
                left['samples'].pop()
            elif kind == 'nonfinite_flops':
                left['samples'][0]['metadata']['polar']['trajectory_metadata']['evaluation']['clm']['flops'] = float('nan')
            elif kind == 'credit':
                right['rollout_data']['advantages'][0] += 1
            elif kind == 'wrong_saved_policy':
                right['rollout_id'] = 0
            try:
                run(left, right)
            except AssertionError:
                passed += 1
            else:
                raise AssertionError('Invalid batch accepted for ' + kind)

    result = {'accepted': True, 'checks': passed, 'synthetic_complete_batch': '8x32',
              'synthetic_samples': 256, 'policy': 1, 'cuda_initialized': torch.cuda.is_initialized(),
              'actual_native_buffer_audit': False, 'new_model_tool_judge_calls': 0}
    Path(__file__).with_name('BATCH_CPU_ACCEPTED.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
