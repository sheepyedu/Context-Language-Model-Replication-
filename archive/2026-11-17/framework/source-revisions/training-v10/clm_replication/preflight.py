"""Check the largest retained native microbatch without an optimizer step."""

import hashlib
import json
from pathlib import Path
import time

import torch
from megatron.core import mpu
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from slime.backends.megatron_utils.data import DataIterator, get_batch, get_data_iterator
from slime.backends.megatron_utils.loss import get_log_probs_and_entropy


def parameter_digest(model):
    digest = hashlib.sha256()
    for chunk in model:
        for name, parameter in chunk.named_parameters():
            digest.update(name.encode())
            digest.update(str(tuple(parameter.shape)).encode())
            digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def check_backward(args, model, store_prefix):
    if store_prefix != 'ref_' or args.start_rollout_id != 0:
        return
    run = Path(args.clm_audit_dir).parent
    owner = json.loads((run / 'CURRENT_OWNER.json').read_bytes())
    rank = mpu.get_tensor_model_parallel_rank()
    directory = Path(owner['directory']) / f'backward-preflight-rank{rank}'
    if (directory / 'ACCEPTED.json').exists():
        return
    directory.mkdir(exist_ok=True)
    with (directory / 'INTENT.json').open('x') as handle:
        json.dump({'created': time.time(), 'owner': owner, 'rank': rank,
                   'operation': 'retained_largest_microbatch_forward_backward_no_optimizer'}, handle)
    assert mpu.get_context_parallel_world_size() == 1
    assert mpu.get_data_parallel_world_size() == 1
    source = run / 'training/rollout-0.pt'
    samples = torch.load(source, weights_only=False)['samples']
    lengths = [len(sample['tokens']) for sample in samples]
    iterators, counts = get_data_iterator(args, model, {
        'total_lengths': lengths, 'dynamic_global_batch_size': len(samples)})
    partitions = iterators[0].micro_batch_indices
    assert sorted(index for group in partitions for index in group) == list(range(len(samples)))
    indices = max(partitions, key=lambda group: sum(lengths[index] for index in group))
    selected = [samples[index] for index in indices]
    data = {'tokens': [torch.tensor(sample['tokens'], dtype=torch.long, device='cuda') for sample in selected],
            'loss_masks': [torch.tensor(sample['loss_mask'], dtype=torch.float32, device='cuda') for sample in selected],
            'total_lengths': [len(sample['tokens']) for sample in selected],
            'response_lengths': [sample['response_length'] for sample in selected]}
    batch = get_batch(DataIterator(data, micro_batch_size=len(selected)),
                      ('tokens', 'loss_masks', 'total_lengths', 'response_lengths',
                       'multimodal_train_inputs', 'max_seq_lens'),
                      args.data_pad_size_multiplier, args.qkv_format, args.allgather_cp)
    def stage(name):
        value = {'stage': name, 'created': time.time(), 'rank': rank,
                 'indices': indices, 'padded_tokens': batch['tokens'].numel(),
                 'original_tokens': sum(data['total_lengths']),
                 'response_lengths': data['response_lengths'],
                 'allocated_bytes': torch.cuda.memory_allocated(),
                 'reserved_bytes': torch.cuda.memory_reserved(),
                 'clock_scope': 'host milestones, GPU operations may overlap'}
        (directory / 'PROGRESS.json').write_text(json.dumps(value, indent=2))
        with (directory / 'EVENTS.jsonl').open('a') as handle:
            handle.write(json.dumps(value) + '\n')
    stage('parameter_digest_before')
    before = parameter_digest(model)
    rng_tracker = get_cuda_rng_tracker()
    tracker_states = {name: state.clone() for name, state in rng_tracker.get_states().items()}
    previous_modes = [chunk.training for chunk in model]
    for chunk in model:
        chunk.zero_grad_buffer()
        chunk.train()
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.enable_grad():
            stage('model_forward')
            logits = model[0](input_ids=batch['tokens'], position_ids=None,
                              attention_mask=None, labels=None,
                              packed_seq_params=batch['packed_seq_params'],
                              loss_mask=batch['full_loss_masks'])
            stage('vocabulary_probabilities')
            _, values = get_log_probs_and_entropy(logits, args=args,
                         unconcat_tokens=batch['unconcat_tokens'],
                         total_lengths=batch['total_lengths'],
                         response_lengths=batch['response_lengths'],
                         with_entropy=True, max_seq_lens=batch.get('max_seq_lens'))
            mask = torch.cat(data['loss_masks'])
            logprob = torch.cat(values['log_probs']).reshape(-1)
            entropy = torch.cat(values['entropy']).reshape(-1)
            assert mask.shape == logprob.shape and mask.sum() > 0
            loss = -(logprob * mask).sum() / mask.sum() - args.entropy_coef * entropy.mean()
            assert torch.isfinite(loss)
            stage('model_backward')
            loss.backward()
            stage('gradient_validation')
            norms = []
            for chunk in model:
                for parameter in chunk.parameters():
                    gradient = getattr(parameter, 'main_grad', parameter.grad)
                    if gradient is not None:
                        assert torch.isfinite(gradient).all()
                        norms.append(float(torch.linalg.vector_norm(gradient)))
            assert norms and sum(norms) > 0
            proof = {'loss': float(loss.detach()), 'gradient_norm_sum': sum(norms),
                     'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                     'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    finally:
        rng_tracker.set_states(tracker_states)
        for chunk, previous in zip(model, previous_modes):
            chunk.zero_grad_buffer()
            chunk.zero_grad(set_to_none=True)
            chunk.train(previous)
    after = parameter_digest(model)
    stage('parameter_digest_after')
    assert before == after
    proof.update(accepted=True, created=time.time(), rank=rank, indices=indices,
                 native_microbatches=counts, padded_tokens=batch['tokens'].numel(),
                 original_tokens=sum(data['total_lengths']), response_lengths=data['response_lengths'],
                 parameter_sha256_before=before, parameter_sha256_after=after,
                 optimizer_steps=0, new_generation_calls=0, RNG_restored=True,
                 gradients_cleared=True, generation_or_tool_replayed=False)
    (directory / 'ACCEPTED.json').write_text(json.dumps(proof, indent=2))
    print('CLM_BACKWARD_PREFLIGHT_ACCEPTED', json.dumps(proof), flush=True)
