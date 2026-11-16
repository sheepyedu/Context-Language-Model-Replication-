"""Compare native vocabulary-parallel chunked probabilities and gradients."""

import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from slime.utils.ppo_utils import calculate_log_probs_and_entropy

local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
dist.init_process_group('nccl')
torch.manual_seed(1234 + local_rank)
assert dist.get_world_size() == 4
original = torch.randn(1153, 2048, device='cuda', dtype=torch.float32)
targets = (torch.arange(1153, device='cuda', dtype=torch.long) * 7) % 8192
rows = []
for entropy in (False, True):
    results, gradients = [], []
    for chunk_size in (-1, 128):
        logits = original.clone().requires_grad_()
        logprob, ent = calculate_log_probs_and_entropy(logits, targets, dist.group.WORLD,
                                                      with_entropy=entropy, chunk_size=chunk_size)
        mask = (torch.arange(1153, device='cuda') % 3 != 0).float()
        loss = -(logprob.squeeze(-1) * mask).sum() / mask.sum()
        if entropy:
            loss = loss - 0.03 * ent.mean()
        loss.backward()
        results.append((logprob.detach(), ent.detach() if ent is not None else None))
        gradients.append(logits.grad.detach())
    torch.testing.assert_close(results[0][0], results[1][0], atol=1e-6, rtol=1e-6)
    if entropy:
        torch.testing.assert_close(results[0][1], results[1][1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(gradients[0], gradients[1], atol=1e-7, rtol=1e-6)
    rows.append({'entropy': entropy,
                 'logprob_max_abs_error': float((results[0][0]-results[1][0]).abs().max()),
                 'gradient_max_abs_error': float((gradients[0]-gradients[1]).abs().max())})
proof = {'accepted': True, 'created': time.time(), 'rank': local_rank,
         'tensor_parallel_world_size': 4, 'rows': rows, 'new_generation_calls': 0,
         'operation': 'native calculate_log_probs_and_entropy chunk equivalence'}
Path(f'/mount-data/CLM-Replication/management/LOGPROB_CHUNKS_TP4_RANK{local_rank}.json').write_text(json.dumps(proof, indent=2))
if local_rank == 0:
    print(json.dumps(proof, indent=2))
dist.destroy_process_group()
