"""Validate the supported GDN backward on the authorized Hopper device."""
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import tilelang
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
from fla.ops.backends import BackendRegistry

torch.manual_seed(20261116)
assert os.environ['CUDA_VISIBLE_DEVICES'] == '4'
assert torch.cuda.get_device_properties(0).major == 9
outdir = Path('/mount-data/CLM-Replication/management/gdn-tilelang-check')
outdir.mkdir(exist_ok=True)
criteria = {'max_abs': 0.05, 'relative_l2': 0.02}
(outdir / 'INTENT.json').write_text(json.dumps({'created':time.time(),'criteria':criteria,'tilelang':tilelang.__version__,'operation':'synthetic_gdn_no_model_generation'}))
results = []
for length in (129, 257):
    # Production TP4 has eight value heads, 128-dimensional key/value heads.
    raw = [F.normalize(torch.randn(1,length,4,128,device='cuda'),dim=-1).repeat_interleave(2,dim=2).to(torch.bfloat16),
           F.normalize(torch.randn(1,length,4,128,device='cuda'),dim=-1).repeat_interleave(2,dim=2).to(torch.bfloat16),
           torch.randn(1,length,8,128,device='cuda',dtype=torch.bfloat16),
           -F.softplus(torch.randn(1,length,8,device='cuda')),
           torch.randn(1,length,8,device='cuda',dtype=torch.bfloat16).sigmoid()]
    kernel = [value.detach().clone().requires_grad_() for value in raw]
    reference = [value.detach().float().clone().requires_grad_() for value in raw]
    start = time.monotonic()
    actual, _ = chunk_gated_delta_rule(*kernel)
    upstream = torch.randn_like(actual)
    (actual * upstream).sum().backward()
    expected, _ = naive_recurrent_gated_delta_rule(reference[0],reference[1],reference[2],reference[4],reference[3])
    (expected * upstream.float()).sum().backward()
    torch.cuda.synchronize()
    comparisons = {}
    for name, value, oracle in [('output',actual,expected)] + [(name,a.grad,b.grad) for name,a,b in zip(['q','k','v','g','beta'],kernel,reference)]:
        difference = value.float() - oracle.float()
        maximum = float(difference.abs().max())
        relative = float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(oracle.float()).clamp_min(1e-12))
        assert torch.isfinite(value).all()
        comparisons[name] = {'max_abs':maximum,'relative_l2':relative}
        assert maximum <= criteria['max_abs'] and relative <= criteria['relative_l2'],(length,name,comparisons[name])
    registry = BackendRegistry._registries['common']
    assert 'common:chunk_bwd_dqkwg:tilelang' in registry._logged
    results.append({'length':length,'comparisons':comparisons,'cold_or_warm_elapsed_seconds':time.monotonic()-start})
    print(json.dumps(results[-1]),flush=True)
proof = {'accepted':True,'created':time.time(),'tilelang_version':tilelang.__version__,'torch_version':torch.__version__,'criteria':criteria,'results':results,'physical_gpu':4,'new_generation_calls':0,'optimizer_steps':0}
(outdir / 'ACCEPTED.json').write_text(json.dumps(proof,indent=2))
print('GDN_TILELANG_ACCEPTED',flush=True)
