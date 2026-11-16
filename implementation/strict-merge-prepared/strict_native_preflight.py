"""Validate retained strict-prefix segments with real native TP4 training gradients."""
from copy import deepcopy
import json
from pathlib import Path
import time

import torch
from megatron.core import mpu
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from slime.backends.megatron_utils import model as native_model
from slime.backends.megatron_utils.data import DataIterator, get_batch
from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
from .preflight import parameter_digest

ROOT = Path('/mount-data/CLM-Replication')
PREPARED = ROOT / 'management/strict-merge-native-prepared-v6'
LIMITS = {'logprob_max_abs': 0.01, 'logprob_rms': 0.002,
          'gradient_relative_l2': 0.02, 'tensor_gradient_abs': 0.00005,
          'tensor_gradient_relative_max': 0.02,
          'loss_abs': 0.0005, 'loss_relative': 0.005,
          'warm_compute_speedup_minimum': 1.05}


def tensor_data(rows, fields):
    result = {'tokens': [torch.tensor(row['tokens'], dtype=torch.long, device='cuda') for row in rows],
              'total_lengths': [len(row['tokens']) for row in rows],
              'response_lengths': [row['response_length'] for row in rows]}
    for key in ['loss_masks', 'rollout_log_probs', 'advantages', 'returns', 'log_probs', 'ref_log_probs']:
        if key in fields:
            dtype = torch.int64 if key == 'loss_masks' else torch.float32
            result[key] = [torch.tensor(values, dtype=dtype, device='cuda') for values in fields[key]]
    return result


def clear_gradients(model):
    for chunk in model:
        chunk.zero_grad_buffer()
        chunk.zero_grad(set_to_none=True)


class CaptureWithoutUpdate:
    """The native pipeline finalizes gradients, then this diagnostic copies them."""
    param_groups = []

    def __init__(self, model, capture):
        self.model = model
        self.capture = capture
        self.gradients = {}
        self.calls = 0
        self.compute_seconds = None
        self.started = None

    def step(self):
        torch.cuda.synchronize()
        self.compute_seconds = time.perf_counter() - self.started
        self.calls += 1
        if self.capture:
            for chunk_index, chunk in enumerate(self.model):
                for name, parameter in chunk.named_parameters():
                    gradient = getattr(parameter, 'main_grad', parameter.grad)
                    if gradient is not None:
                        self.gradients[f'{chunk_index}:{name}'] = gradient.detach().float().cpu().clone()
        return True, 1.0, 0

    def zero_grad(self):
        for chunk in self.model:
            chunk.zero_grad(set_to_none=True)


class NoScheduleUpdate:
    def step(self, increment):
        pass


def native_backward(args, model, rows, fields, capture):
    clear_gradients(model)
    data = tensor_data(rows, fields)
    optimizer = CaptureWithoutUpdate(model, capture)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    optimizer.started = time.perf_counter()
    loss, _ = native_model.train_one_step(args, 0, 0,
        [DataIterator(data, micro_batch_size=1)], model, optimizer, NoScheduleUpdate(), len(rows))
    assert optimizer.calls == 1
    result = {'loss': loss, 'compute_seconds': optimizer.compute_seconds,
              'peak_allocated': torch.cuda.max_memory_allocated(),
              'peak_reserved': torch.cuda.max_memory_reserved(),
              'diagnostic_capture_callbacks': optimizer.calls,
              'actual_optimizer_updates': 0}
    return result, optimizer.gradients


def reference_logprobs(args, model, rows, fields):
    data = tensor_data(rows, fields)
    iterator = DataIterator(data, micro_batch_size=1)
    values = []
    with torch.no_grad():
        for _ in rows:
            batch = get_batch(iterator,
                ['tokens', 'loss_masks', 'total_lengths', 'response_lengths',
                 'multimodal_train_inputs', 'max_seq_lens'],
                args.data_pad_size_multiplier, args.qkv_format, args.allgather_cp)
            logits = model[0](input_ids=batch['tokens'], position_ids=None,
                attention_mask=None, labels=None, packed_seq_params=batch['packed_seq_params'],
                loss_mask=batch['full_loss_masks'])
            _, output = get_log_probs_and_entropy(logits, args=args,
                unconcat_tokens=batch['unconcat_tokens'], total_lengths=batch['total_lengths'],
                response_lengths=batch['response_lengths'], with_entropy=False,
                max_seq_lens=batch.get('max_seq_lens'))
            values.extend(item.detach().cpu().tolist() for item in output['log_probs'])
    return values


def merge_reference(case, original):
    row = case['merged_rows'][0]
    offset = len(row['tokens']) - row['response_length']
    merged = [0.0] * row['response_length']
    for index, span in enumerate(case['spans']):
        start = span['generated_start'] - offset
        merged[start:start + span['generated_length']] = original[index]
    return [merged]


def compare_gradients(original, merged):
    assert original.keys() == merged.keys()
    squared_original = 0.0
    squared_difference = 0.0
    tensors = []
    for name in original:
        a, b = original[name], merged[name]
        assert a.shape == b.shape
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        delta = a - b
        maximum = float(delta.abs().max())
        scale = float(a.abs().max())
        tolerance = LIMITS['tensor_gradient_abs'] + LIMITS['tensor_gradient_relative_max'] * scale
        squared_original += float(torch.sum(a.double().square()))
        squared_difference += float(torch.sum(delta.double().square()))
        tensors.append({'name': name, 'numel': a.numel(), 'max_abs_diff': maximum,
                        'original_max_abs': scale, 'tolerance': tolerance,
                        'accepted': maximum <= tolerance})
    assert squared_original > 0
    relative = (squared_difference / squared_original) ** 0.5
    return {'accepted': relative <= LIMITS['gradient_relative_l2'] and all(row['accepted'] for row in tensors),
            'relative_l2': relative, 'squared_original': squared_original,
            'squared_difference': squared_difference, 'tensors': tensors}


def check_equivalence(args, model, store_prefix):
    assert store_prefix == 'ref_' and args.start_rollout_id == 0
    assert mpu.get_tensor_model_parallel_world_size() == 4
    assert mpu.get_context_parallel_world_size() == 1
    assert mpu.get_data_parallel_world_size() == 1
    assert not args.overlap_grad_reduce and not args.overlap_param_gather
    assert args.check_for_nan_in_loss_and_grad and not args.fp16
    rank = mpu.get_tensor_model_parallel_rank()
    destination = PREPARED / f'native-rank-{rank}'
    destination.mkdir(exist_ok=False)
    with (destination / 'INTENT.json').open('x') as handle:
        json.dump({'rank': rank, 'created': time.time(), 'limits': LIMITS,
                   'actual_optimizer_updates': 0, 'input': 'FIXTURE_CASES.pt'}, handle, indent=2)
    cases = torch.load(PREPARED / 'FIXTURE_CASES.pt', weights_only=False)['cases']
    tracker = get_cuda_rng_tracker()
    tracker_states = {key: value.clone() for key, value in tracker.get_states().items()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    modes = [chunk.training for chunk in model]
    config = native_model.get_model_config(model[0])
    config_fields = ['grad_scale_func', 'timers', 'finalize_model_grads_func']
    original_config = {key: getattr(config, key) for key in config_fields}
    before = parameter_digest(model)
    results = []

    def reset_rng():
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        tracker.set_states({key: value.clone() for key, value in tracker_states.items()})

    try:
        for chunk in model:
            chunk.train()
        config.grad_scale_func = lambda loss: loss
        config.timers = None
        config.finalize_model_grads_func = native_model.finalize_model_grads
        with torch.enable_grad():
            for case in cases:
                reset_rng()
                original_lp = reference_logprobs(args, model, case['original_rows'], case['original_fields'])
                reset_rng()
                merged_lp = reference_logprobs(args, model, case['merged_rows'], case['merged_fields'])
                expected = merge_reference(case, original_lp)[0]
                mask = torch.tensor(case['merged_fields']['loss_masks'][0], dtype=torch.bool)
                lp_difference = torch.tensor(merged_lp[0])[mask] - torch.tensor(expected)[mask]
                logprob_maximum = float(lp_difference.abs().max())
                logprob_rms = float(lp_difference.square().mean().sqrt())
                original_fields = deepcopy(case['original_fields'])
                merged_fields = deepcopy(case['merged_fields'])
                # Fixed diagnostic offsets exercise a nonzero reference KL and old-policy ratio.
                original_fields['log_probs'] = [[value + .015 for value in values] for values in original_lp]
                original_fields['ref_log_probs'] = [[value + .012 for value in values] for values in original_lp]
                merged_fields['log_probs'] = merge_reference(case, original_fields['log_probs'])
                merged_fields['ref_log_probs'] = merge_reference(case, original_fields['ref_log_probs'])
                reset_rng()
                first, original_grad = native_backward(args, model, case['original_rows'], original_fields, True)
                reset_rng()
                second, merged_grad = native_backward(args, model, case['merged_rows'], merged_fields, True)
                comparison = compare_gradients(original_grad, merged_grad)
                del original_grad, merged_grad
                loss_error = abs(first['loss']['loss'] - second['loss']['loss'])
                loss_limit = LIMITS['loss_abs'] + LIMITS['loss_relative'] * abs(first['loss']['loss'])
                reset_rng()
                warm_original, _ = native_backward(args, model, case['original_rows'], original_fields, False)
                reset_rng()
                warm_merged, _ = native_backward(args, model, case['merged_rows'], merged_fields, False)
                accepted = (comparison['accepted'] and logprob_maximum <= LIMITS['logprob_max_abs']
                            and logprob_rms <= LIMITS['logprob_rms'] and loss_error <= loss_limit)
                result = {'case_id': case['case_id'], 'accepted': accepted,
                          'logprob_max_abs': logprob_maximum, 'logprob_rms': logprob_rms,
                          'loss_abs_diff': loss_error, 'loss_tolerance': loss_limit,
                          'gradient_comparison': comparison, 'original': first, 'merged': second,
                          'warm_original': warm_original, 'warm_merged': warm_merged}
                results.append(result)
                (destination / f"case-{case['case_id']}.json").write_text(json.dumps(result, indent=2))
                collective = torch.tensor(int(accepted), dtype=torch.int32, device='cuda')
                torch.distributed.all_reduce(collective, op=torch.distributed.ReduceOp.MIN)
                if not bool(collective.item()):
                    break
    finally:
        reset_rng()
        clear_gradients(model)
        for chunk, previous in zip(model, modes):
            chunk.train(previous)
        for key, value in original_config.items():
            setattr(config, key, value)
    after = parameter_digest(model)
    assert before == after
    speedup = sum(row['warm_original']['compute_seconds'] for row in results) / sum(
        row['warm_merged']['compute_seconds'] for row in results)
    proof = {'accepted': len(results) == len(cases) and all(row['accepted'] for row in results) and speedup >= LIMITS['warm_compute_speedup_minimum'],
             'rank': rank, 'created': time.time(), 'cases': [row['case_id'] for row in results],
             'warm_compute_speedup': speedup, 'limits': LIMITS,
             'parameter_sha256_before': before, 'parameter_sha256_after': after,
             'actual_optimizer_updates': 0, 'diagnostic_capture_callbacks_are_parameter_updates': False,
             'RNG_restored': True, 'gradients_cleared': True, 'new_generation_calls': 0}
    (destination / 'RESULT.json').write_text(json.dumps(proof, indent=2))
    torch.distributed.barrier()
    # This isolated fixture terminates before the production reference or optimizer loop.
    raise RuntimeError('CLM_STRICT_MERGE_NATIVE_FIXTURE_FINISHED_NO_OPTIMIZER')
