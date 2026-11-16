"""Exercise the native offload optimizer backend on a tiny isolated tensor."""
from io import BytesIO
import json
import os
from pathlib import Path
import time

import torch
from torch.optim import AdamW
from transformer_engine.pytorch.optimizers import FusedAdam
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

root=Path('/mount-data/CLM-Replication')
assert os.environ['CUDA_VISIBLE_DEVICES']=='4'
# Current production is still reference-forward, with no model parameters touched here.
log=(root/'runs/c2-seed1234/attempt-11/controller.log').read_text(errors='replace')[-8192:]
assert 'ref forward:' in log and 'train/grad_norm' not in log
assert torch.cuda.mem_get_info()[0]>4*1024**3
torch.manual_seed(20261116)
initial=torch.randn(4096,dtype=torch.float32)*0.02
parameter=torch.nn.Parameter(initial.cuda())
reference=torch.nn.Parameter(initial.clone())
kwargs=dict(lr=1e-6,weight_decay=0.1,betas=(0.9,0.98),eps=1e-8)
optimizer=HybridDeviceOptimizer([parameter],offload_fraction=1.0,cpu_optimizer_cls=AdamW,
    gpu_optimizer_cls=FusedAdam,overlap_cpu_optimizer_d2h_h2d=False,pin_cpu_grads=True,
    pin_cpu_params=True,param_update_in_fp32=True,bias_correction=True,fused=True,**kwargs)
control=AdamW([reference],fused=True,**kwargs)
maximum=0
for step in range(3):
    gradient=torch.sin(torch.arange(4096,dtype=torch.float32)+step)
    parameter.decoupled_grad=gradient.cuda()
    reference.grad=gradient
    optimizer.step();control.step();torch.cuda.synchronize()
    torch.testing.assert_close(parameter.detach().cpu(),reference.detach(),atol=1e-8,rtol=1e-6)
    maximum=max(maximum,float((parameter.detach().cpu()-reference.detach()).abs().max()))
    optimizer.zero_grad();control.zero_grad()
assert (parameter.detach().cpu()!=initial).any()
state=BytesIO();torch.save(optimizer.state_dict(),state);state.seek(0)
assert torch.load(state,weights_only=False)
proof={'accepted':True,'created':time.time(),'physical_gpu':4,'optimizer_class':'native HybridDeviceOptimizer',
       'cpu_class':'torch.optim.AdamW fused=True','offload_fraction':1.0,'pin_cpu_grads':True,
       'pin_cpu_params':True,'overlap':False,'hyperparameters':kwargs,'synthetic_backend_updates':3,
       'parameter_count':4096,'reference_max_abs_error':maximum,'state_serialization_bytes':state.tell(),
       'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
       'formal_model_parameters_touched':False,'formal_optimizer_updates':0,'new_generation_calls':0,
       'distributed_optimizer_checkpoint_restore':'pending'}
(root/'management/HYBRID_OPTIMIZER_BACKEND_ACCEPTED.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof,indent=2))
