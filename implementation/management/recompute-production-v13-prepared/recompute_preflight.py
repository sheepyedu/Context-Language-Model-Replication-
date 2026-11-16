"""Compare activation checkpoint groups on the same retained native microbatch."""
import json
from pathlib import Path
import time

import torch
from megatron.core import mpu
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from slime.backends.megatron_utils.data import DataIterator, get_batch, get_data_iterator
from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
from slime.backends.megatron_utils.model import get_model_config
from .preflight import parameter_digest


def check_backward(args, model, store_prefix):
    if store_prefix != 'ref_' or args.start_rollout_id != 0:
        return
    assert args.recompute_num_layers == 4
    assert mpu.get_data_parallel_world_size() == 1
    assert mpu.get_context_parallel_world_size() == 1
    assert mpu.get_tensor_model_parallel_world_size() == 4
    run = Path(args.clm_audit_dir).parent
    owner = json.loads((run / 'CURRENT_OWNER.json').read_text())
    rank = mpu.get_tensor_model_parallel_rank()
    directory = Path(owner['directory']) / f'recompute-preflight-rank{rank}'
    directory.mkdir(exist_ok=False)
    (directory / 'INTENT.json').write_text(json.dumps({
        'created': time.time(), 'rank': rank, 'groups': [1, 4],
        'optimizer_updates': 0, 'new_generation_calls': 0}, indent=2))
    samples = torch.load(run / 'training/rollout-0.pt', weights_only=False)['samples']
    lengths = [len(sample['tokens']) for sample in samples]
    iterators, counts = get_data_iterator(args, model, {
        'total_lengths': lengths, 'dynamic_global_batch_size': len(samples)})
    partitions = iterators[0].micro_batch_indices
    indices = max(partitions, key=lambda group: sum(lengths[index] for index in group))
    selected = [samples[index] for index in indices]
    data = {'tokens': [torch.tensor(s['tokens'], dtype=torch.long, device='cuda') for s in selected],
            'loss_masks': [torch.tensor(s['loss_mask'], dtype=torch.float32, device='cuda') for s in selected],
            'total_lengths': [len(s['tokens']) for s in selected],
            'response_lengths': [s['response_length'] for s in selected]}
    batch = get_batch(DataIterator(data, micro_batch_size=len(selected)),
        ('tokens', 'loss_masks', 'total_lengths', 'response_lengths',
         'multimodal_train_inputs', 'max_seq_lens'),
        args.data_pad_size_multiplier, args.qkv_format, args.allgather_cp)
    config = get_model_config(model[0])
    assert config.recompute_num_layers == 4
    before = parameter_digest(model)
    tracker = get_cuda_rng_tracker()
    tracker_states = {k: v.clone() for k, v in tracker.get_states().items()}
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    modes = [chunk.training for chunk in model]

    def reset():
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        tracker.set_states({k: v.clone() for k, v in tracker_states.items()})
        for chunk in model:
            chunk.zero_grad_buffer()
            chunk.zero_grad(set_to_none=True)
            chunk.train()

    gradients = {}
    results = []
    differences = []
    try:
        for group in [1, 4]:
            reset()
            config.recompute_num_layers = group
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            (directory / 'PROGRESS.json').write_text(json.dumps({
                'group': group, 'stage': 'native_forward_backward',
                'indices': indices, 'padded_tokens': batch['tokens'].numel(),
                'created': time.time()}))
            with torch.enable_grad():
                logits = model[0](input_ids=batch['tokens'], position_ids=None,
                    attention_mask=None, labels=None, packed_seq_params=batch['packed_seq_params'],
                    loss_mask=batch['full_loss_masks'])
                _, output = get_log_probs_and_entropy(logits, args=args,
                    unconcat_tokens=batch['unconcat_tokens'], total_lengths=batch['total_lengths'],
                    response_lengths=batch['response_lengths'], with_entropy=False,
                    max_seq_lens=batch.get('max_seq_lens'))
                mask = torch.cat(data['loss_masks'])
                lp = torch.cat(output['log_probs']).reshape(-1)
                loss = -(lp * mask).sum() / mask.sum()
                assert torch.isfinite(loss)
                loss.backward()
            torch.cuda.synchronize()
            result = {'group': group, 'loss': float(loss.detach()),
                'forward_backward_seconds': time.perf_counter() - started,
                'peak_allocated': torch.cuda.max_memory_allocated(),
                'peak_reserved': torch.cuda.max_memory_reserved()}
            for chunk_index, chunk in enumerate(model):
                for name, parameter in chunk.named_parameters():
                    grad = getattr(parameter, 'main_grad', parameter.grad)
                    if grad is None:
                        continue
                    assert torch.isfinite(grad).all()
                    key = f'{chunk_index}:{name}'
                    current = grad.detach().float().cpu().clone()
                    if group == 1:
                        gradients[key] = current
                    else:
                        original = gradients.pop(key)
                        maximum = float((current - original).abs().max())
                        scale = float(original.abs().max())
                        # Fixed tight comparison for changing only recomputation boundaries.
                        tolerance = 1e-6 + 1e-4 * scale
                        differences.append({'name': key, 'max_abs': maximum,
                            'original_max_abs': scale, 'tolerance': tolerance,
                            'accepted': maximum <= tolerance})
            results.append(result)
            del logits, output, lp, loss
        assert not gradients and differences
        accepted = all(x['accepted'] for x in differences)
        accepted = accepted and abs(results[0]['loss'] - results[1]['loss']) <= 1e-6
    finally:
        config.recompute_num_layers = 4
        reset()
        for chunk, mode in zip(model, modes):
            chunk.train(mode)
    after = parameter_digest(model)
    assert before == after
    proof = {'accepted': accepted, 'created': time.time(), 'rank': rank,
        'indices': indices, 'padded_tokens': batch['tokens'].numel(),
        'native_microbatches': counts, 'cases': results, 'gradients': differences,
        'parameter_sha256_before': before, 'parameter_sha256_after': after,
        'RNG_restored': True, 'gradients_cleared': True, 'optimizer_updates': 0,
        'new_generation_calls': 0, 'science_samples_merged': False}
    (directory / 'RESULT.json').write_text(json.dumps(proof, indent=2))
    flag = torch.tensor(int(accepted), dtype=torch.int32, device='cuda')
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    assert flag.item(), 'Activation checkpoint grouping gradient comparison failed'
    print('CLM_RECOMPUTE_PREFLIGHT_ACCEPTED', rank, flush=True)
