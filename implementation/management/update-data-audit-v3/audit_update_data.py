"""Read the completed native training buffer and original rollout on CPU."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import time

import torch




def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def check_row(row, sample, expected):
    length = sample['response_length']
    assert row['response_length'] == length
    assert row['sample_index'] == sample['index']
    assert torch.equal(row['tokens'], torch.tensor(sample['tokens'], dtype=torch.int64))
    assert row['total_length'] == len(sample['tokens'])
    assert torch.equal(row['loss_mask'], torch.tensor(sample['loss_mask']))
    assert torch.equal(row['rollout_log_probs'], torch.tensor(sample['rollout_log_probs']))
    assert math.isclose(row['raw_reward'], expected['raw'], rel_tol=0, abs_tol=1e-12)
    assert math.isclose(row['reward'], expected['task'], rel_tol=0, abs_tol=1e-12)
    assert math.isclose(row['efficiency'], expected['efficiency'], rel_tol=0, abs_tol=1e-12)
    fields = ('loss_mask', 'role_mask', 'rollout_log_probs', 'ref_log_probs',
              'kl', 'advantages', 'returns')
    for field in fields:
        tensor = row[field]
        assert tensor.ndim == 1 and len(tensor) == length, field
        assert bool(torch.isfinite(tensor).all()), field
    assert bool(((row['loss_mask'] == 0) | (row['loss_mask'] == 1)).all())
    assert bool(((row['role_mask'] == 0) | (row['role_mask'] == 1)).all())
    assert bool((row['role_mask'] <= row['loss_mask']).all())
    assert bool((row['role_mask'] == expected['role']).all())
    task_credit = torch.full_like(row['advantages'], expected['task'])
    expected_credit = task_credit + 0.25 * expected['efficiency'] * row['role_mask']
    assert torch.equal(row['advantages'], expected_credit)
    assert torch.equal(row['returns'], expected_credit)


def native_identity(pid):
    process = Path('/proc') / str(pid)
    return {'pid': pid, 'uid': process.stat().st_uid,
            'start_ticks': int((process / 'stat').read_text().split()[21]),
            'raw_sha256': digest(process / 'cmdline')}


def audit(run, rollout_id, expected_owner):
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    if not isinstance(rollout_id, int) or isinstance(rollout_id, bool) or rollout_id < 0:
        raise ValueError('A nonnegative native rollout index is required')
    owner_before = native_identity(expected_owner['pid'])
    assert owner_before == expected_owner
    run = Path(run)
    paths = {'rollout': run / f'rollout-{rollout_id}.pt',
             'train': run / f'train-{rollout_id}.pt'}
    before = {name: (path.stat().st_size, path.stat().st_mtime_ns)
              for name, path in paths.items()}
    rollout = torch.load(paths['rollout'], map_location='cpu', weights_only=False)
    saved = torch.load(paths['train'], map_location='cpu', weights_only=False)
    assert rollout['rollout_id'] == saved['rollout_id'] == rollout_id
    samples = rollout['samples']
    data = saved['rollout_data']
    assert len(samples) == data['dynamic_global_batch_size'] and len(samples) > 0
    groups = defaultdict(dict)
    for sample in samples:
        polar = sample['metadata']['polar']
        outcome = polar['trajectory_metadata']['evaluation']['clm']
        assert polar['trajectory_metadata']['policy_version'] == rollout_id
        key = (sample['group_index'], polar['session_id'])
        known = groups[sample['group_index']].setdefault(key, outcome)
        assert known == outcome
    assert len(groups) == 8
    assert all(len(group) == 32 for group in groups.values())
    expected_by_trajectory = {}
    group_summaries = []
    for group_index, group in sorted(groups.items()):
        raw = {key: (1.0 if value['success'] else
                     -0.3 * value['no_tool'] - 0.2 * value['format_failure'])
               for key, value in group.items()}
        mean = statistics.mean(raw.values())
        std = statistics.stdev(raw.values())
        assert std > 0
        successful_costs = [value['flops'] for value in group.values()
                            if value['success'] and value['clean_finish']]
        mean_cost = statistics.mean(successful_costs) if len(successful_costs) >= 2 else None
        for key, outcome in group.items():
            assert math.isfinite(outcome['flops']) and outcome['flops'] > 0
            efficiency = 0.0
            if mean_cost is not None and outcome['success'] and outcome['clean_finish']:
                efficiency = max(-1.0, min(1.0, (mean_cost - outcome['flops']) / mean_cost))
            expected_by_trajectory[key] = {'raw': raw[key],
                                           'task': (raw[key] - mean) / (std + 1e-6),
                                           'efficiency': efficiency}
        group_summaries.append({'group_index': group_index, 'trajectories': len(group),
                                'successes': sum(value['success'] for value in group.values()),
                                'rewards': dict(Counter(map(str, raw.values()))),
                                'task_reward_std': std})
    mapping = {'tokens': 'tokens', 'response_length': 'response_lengths',
               'sample_index': 'sample_indices', 'total_length': 'total_lengths',
               'loss_mask': 'loss_masks', 'rollout_log_probs': 'rollout_log_probs',
               'role_mask': 'role_masks', 'efficiency': 'a_eff', 'reward': 'rewards',
               'raw_reward': 'raw_reward', 'ref_log_probs': 'ref_log_probs',
               'kl': 'kl', 'advantages': 'advantages', 'returns': 'returns'}
    assert all(len(data[key]) == len(samples) for key in mapping.values())
    counts = Counter()
    for index, sample in enumerate(samples):
        polar = sample['metadata']['polar']
        expected = dict(expected_by_trajectory[(sample['group_index'], polar['session_id'])])
        outcome = polar['trajectory_metadata']['evaluation']['clm']
        expected['role'] = int(outcome['pure_edit_by_call'][polar['trace_index']])
        row = {key: data[value][index] for key, value in mapping.items()}
        check_row(row, sample, expected)
        counts['samples'] += 1
        counts['input_token_occurrences'] += len(sample['tokens']) - sample['response_length']
        counts['output_tokens'] += sample['response_length']
        counts['loss_tokens'] += int(row['loss_mask'].sum())
        counts['edit_role_tokens'] += int(row['role_mask'].sum())
        if expected['efficiency'] != 0:
            counts['nonzero_efficiency_role_tokens'] += int(row['role_mask'].sum())
    inputs = {name: {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}
              for name, path in paths.items()}
    assert before == {name: (path.stat().st_size, path.stat().st_mtime_ns)
                      for name, path in paths.items()}
    owner_after = native_identity(expected_owner['pid'])
    assert owner_before == owner_after
    return {'accepted': True, 'created': time.time(), 'rollout_id': rollout_id,
            'logical_policy_before_update': rollout_id,
            'prospective_formal_update': rollout_id + 1,
            'parameter_update_accepted': False,
            'logical_trajectories': len(expected_by_trajectory), 'groups': group_summaries,
            'counts': dict(counts), 'inputs': inputs, 'owner': owner_after,
            'saved_rank': saved['rank'], 'scope': 'one actually retained native rank buffer and all original rollout samples',
            'other_rank_buffers': 'not separately retained by current filename template',
            'checkpoint_restore_accepted': False, 'three_update_stability_accepted': False,
            'cuda_initialized': torch.cuda.is_initialized(), 'new_model_tool_judge_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--rollout-id', required=True, type=int)
    parser.add_argument('--owner-pid', required=True, type=int)
    parser.add_argument('--owner-start', required=True, type=int)
    parser.add_argument('--owner-raw-sha256', required=True)
    parser.add_argument('--owner-uid', type=int, default=0)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    signal.alarm(300)
    started = time.time()
    expected_owner = {'pid': args.owner_pid, 'uid': args.owner_uid,
                      'start_ticks': args.owner_start,
                      'raw_sha256': args.owner_raw_sha256}
    try:
        result = audit(args.run, args.rollout_id, expected_owner)
        result['elapsed_seconds'] = time.time() - started
        (args.output / 'ACCEPTED.json').write_text(json.dumps(result, indent=2))
        exit_code = 0
    except BaseException as error:
        result = {'accepted': False, 'created': time.time(), 'error': type(error).__name__,
                  'message': str(error)}
        (args.output / 'FAILURE.json').write_text(json.dumps(result, indent=2))
        exit_code = 1
    (args.output / 'EXIT.json').write_text(json.dumps({'exit_code': exit_code, 'created': time.time()}))
    print(json.dumps(result), flush=True)
    os._exit(exit_code)
