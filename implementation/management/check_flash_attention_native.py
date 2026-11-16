"""Verify native THD padding and D=256 GQA forward/backward against math SDPA."""

import hashlib
import json
import os
from pathlib import Path
import time

os.environ.update(NVTE_FLASH_ATTN='1', NVTE_FUSED_ATTN='0', NVTE_UNFUSED_ATTN='0')
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformer_engine.pytorch import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils

assert FlashAttentionUtils.is_installed
assert str(FlashAttentionUtils.version) == '2.8.3'
assert torch.cuda.device_count() == 2
torch.cuda.set_device(0)
torch.manual_seed(1234)
q = torch.randn(64, 8, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(64, 2, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(64, 2, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
# Native Slime adds trailing padding as an independent, zero-loss sequence.
cu = torch.tensor([0, 17, 46, 64], device='cuda', dtype=torch.int32)
attention = DotProductAttention(8, 256, num_gqa_groups=2, qkv_format='thd',
                                attn_mask_type='padding_causal', attention_dropout=0)
actual = attention(q, k, v, cu_seqlens_q=cu, cu_seqlens_kv=cu,
                   max_seqlen_q=29, max_seqlen_kv=29).reshape(64, 8, 256)[:46]
references = []
with sdpa_kernel(SDPBackend.MATH):
    for offset, length in ((0, 17), (17, 29)):
        query = q[offset:offset+length].float().transpose(0, 1).unsqueeze(0)
        key = k[offset:offset+length].float().transpose(0, 1).unsqueeze(0)
        value = v[offset:offset+length].float().transpose(0, 1).unsqueeze(0)
        references.append(F.scaled_dot_product_attention(query, key, value,
                          is_causal=True, enable_gqa=True).squeeze(0).transpose(0, 1))
expected = torch.cat(references)
torch.testing.assert_close(actual.float(), expected, atol=0.03, rtol=0.03)
actual.float().square().sum().backward()
gradients = [tensor.grad.clone().float() for tensor in (q, k, v)]
for tensor in (q, k, v):
    tensor.grad = None
expected.square().sum().backward()
gradient_errors = []
for tensor, gradient in zip((q, k, v), gradients):
    reference = tensor.grad.float()
    assert torch.isfinite(gradient).all()
    assert all(gradient[:46, :, start:start+64].norm() > 0 for start in (0, 64, 128, 192))
    torch.testing.assert_close(gradient, reference, atol=0.25, rtol=0.06)
    gradient_errors.append(float((gradient-reference).abs().max()))
proof = {'accepted': True, 'created': time.time(), 'pid': os.getpid(),
         'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
         'flash_attention_version': str(FlashAttentionUtils.version),
         'torch_version': torch.__version__, 'cudnn_version': torch.backends.cudnn.version(),
         'layout': 'native THD, BF16 GQA 8/2, D256, independent trailing padding sequence',
         'reference': 'float32 PyTorch SDPA forced MATH backend',
         'forward_max_abs_error': float((actual.detach().float()-expected.detach()).abs().max()),
         'gradient_max_abs_errors': gradient_errors, 'all_D256_gradient_channels_nonzero': True,
         'new_generation_calls': 0}
Path('/mount-data/CLM-Replication/management/FLASH_NATIVE_ROOT_ACCEPTED.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof, indent=2))
