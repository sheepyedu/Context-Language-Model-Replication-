"""CPU-only native integration and causal gradient checks for strict segments."""
from argparse import Namespace
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import sys
import time

import torch
import torch.nn.functional as F
from slime.utils.types import Sample
import slime.ray.rollout as native_rollout
from slime.rollout.base_types import RolloutFnTrainOutput
from clm_replication import bridge

ROOT = Path('/mount-data/CLM-Replication')
PREPARED = ROOT / 'management/strict-merge-candidate-v2'
for name in ['strict_merge', 'strict_training']:
    spec=importlib.util.spec_from_file_location('clm_replication.'+name, PREPARED/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
from clm_replication.strict_training import merge_prepared_samples, post_process_rewards

assert torch.cuda.device_count()==0
torch.set_num_threads(8)
args=Namespace(**json.loads((ROOT/'runs/c2-seed1234/attempt-11/effective_args.json').read_text()))
samples=[Sample.from_dict(item) for item in torch.load(ROOT/'runs/c2-seed1234/training/rollout-0.pt',weights_only=False)['samples']]
assert len(samples)==11095
with tempfile.TemporaryDirectory() as directory:
    args.clm_audit_dir=directory
    raw, advantages=bridge.post_process_rewards(args,samples)
merged,spans=merge_prepared_samples(samples,raw,advantages)
raw_merged,advantages_merged=post_process_rewards(args,merged)
manager_type=native_rollout.RolloutManager.__ray_metadata__.modified_class
manager=SimpleNamespace(args=args,custom_convert_samples_to_train_data_func=None,
                        _post_process_rewards=lambda items:(raw,advantages))
original=manager_type._convert_samples_to_train_data(manager,samples)
manager._post_process_rewards=lambda items:post_process_rewards(args,items)
converted=manager_type._convert_samples_to_train_data(manager,merged)
assert sum(map(sum,original['loss_masks']))==sum(map(sum,converted['loss_masks']))
assert sum(map(sum,original['role_masks']))==sum(map(sum,converted['role_masks']))
for position,sources in enumerate(spans):
    offset=len(merged[position].tokens)-merged[position].response_length
    for source in sources:
        old=source['source_index'];start=source['generated_start']-offset;end=start+source['generated_length']
        for field in ['loss_masks','role_masks','rollout_log_probs']:
            assert converted[field][position][start:end]==original[field][old]
        for field in ['rewards','raw_reward','a_eff','truncated']:
            assert converted[field][position]==original[field][old]
# Native admission sizes the merged samples before conversion, retaining one update.
manager.train_parallel_config={'dp_size':1}
manager.data_source=None
manager.generate_rollout=lambda *args,**kwargs:RolloutFnTrainOutput(samples=merged,metrics={})
manager._compute_dynamic_global_batch_size=MethodType(manager_type._compute_dynamic_global_batch_size,manager)
sized,_=manager_type._get_rollout_data(manager,0)
assert len(sized)==len(merged)==manager._dynamic_global_batch_size
assert len(sized)//manager._dynamic_global_batch_size==1
# Exercise the actual DP1 splitter without a Ray connection.
original_put=native_rollout.ray.put
try:
    native_rollout.ray.put=lambda value:value
    partitions=manager_type._split_train_data_by_dp(manager,converted,1)
finally:
    native_rollout.ray.put=original_put
partition=partitions[0].inner
assert partition['dynamic_global_batch_size']==len(merged)
assert partition['partition']==range(len(merged))
for field in ['tokens','loss_masks','role_masks','rewards','a_eff','rollout_log_probs']:
    assert partition[field]==converted[field]
# Use the actual unchanged dual-channel hook on both views.
from clm_replication.advantages import custom_advantage_function
def advantages_for(data):
    view={**data,'kl':[torch.zeros(length) for length in data['response_lengths']]}
    custom_advantage_function(args,view)
    return view['advantages']
original_advantages=advantages_for(original)
merged_advantages=advantages_for(converted)
for position,sources in enumerate(spans):
    offset=len(merged[position].tokens)-merged[position].response_length
    for source in sources:
        start=source['generated_start']-offset
        end=start+source['generated_length']
        torch.testing.assert_close(merged_advantages[position][start:end],
                    original_advantages[source['source_index']],atol=0,rtol=0)

# The prepared view survives the native debug sample serialization.
roundtrip=[Sample.from_dict(item.to_dict()) for item in merged[:2]]
assert post_process_rewards(args,roundtrip)==(raw_merged[:2],advantages_merged[:2])
corrupted=deepcopy(merged[0]);corrupted.loss_mask[0]=1-corrupted.loss_mask[0]
try:
    post_process_rewards(args,[corrupted])
    raise AssertionError('Tampered prepared view was accepted')
except ValueError:
    pass

# Parameter gradients for an actual causal attention module, with gap tokens masked.
class CausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=torch.nn.Embedding(64,16,dtype=torch.float64)
        self.qkv=torch.nn.Linear(16,48,bias=False,dtype=torch.float64)
        self.head=torch.nn.Linear(16,64,bias=False,dtype=torch.float64)
    def forward(self,tokens):
        hidden=self.embedding(torch.tensor(tokens))
        q,k,v=self.qkv(hidden).chunk(3,dim=-1)
        hidden=F.scaled_dot_product_attention(q[None,None],k[None,None],v[None,None],is_causal=True)[0,0]
        return self.head(hidden)

def synthetic(tokens,response_length,index):
    return Sample(group_index=0,index=index,tokens=tokens,response_length=response_length,
          loss_mask=[1]*response_length,rollout_log_probs=[-3.5]*response_length,
          metadata={'a_eff':0.15,'polar':{'session_id':'synthetic','trace_index':index,
                    'role_mask':[index%2]*response_length,'trajectory_metadata':{'policy_version':0}}})
fixture=[synthetic([1,2,3,4,5],2,0),synthetic([1,2,3,4,5,20,21,6,7],2,1),
         synthetic([1,2,3,4,5,20,21,6,7,22,8,9],2,2)]
joined,_=merge_prepared_samples(fixture,[1]*3,[0.8]*3)
assert len(joined)==1

def objective(model,items):
    loss=torch.zeros((),dtype=torch.float64)
    denominator=sum(sum(item.loss_mask) for item in items)
    values=[]
    for item in items:
        start=len(item.tokens)-item.response_length
        logits=model(item.tokens) / 0.7
        selected=logits[start-1:-1]
        targets=torch.tensor(item.tokens[start:])
        log_probs=selected.log_softmax(-1).gather(-1,targets[:,None]).squeeze(-1)
        # Frozen old/reference values represent the exact same original causal inputs.
        old=log_probs.detach()+0.015
        reference=log_probs.detach()-0.012
        role=torch.tensor(item.metadata['polar']['role_mask'])
        adv=0.8+0.25*0.15*role
        ratio=(log_probs-old).exp()
        pg=torch.maximum(-adv*ratio,-adv*ratio.clamp(0.8,1.28))
        importance=(log_probs.detach()-torch.tensor(item.rollout_log_probs)).exp().clamp(0,2)
        delta=reference-log_probs
        kl=delta.exp()-delta-1
        mask=torch.tensor(item.loss_mask)
        loss=loss+((pg*importance+0.01*kl)*mask).sum()/denominator
        values.extend(log_probs[mask.bool()].detach().tolist())
    return loss,torch.tensor(values)

torch.manual_seed(20261116)
a=CausalModel();b=deepcopy(a)
loss_a,values_a=objective(a,fixture);loss_b,values_b=objective(b,joined)
loss_a.backward();loss_b.backward()
torch.testing.assert_close(values_a,values_b,atol=1e-12,rtol=1e-12)
torch.testing.assert_close(loss_a,loss_b,atol=1e-12,rtol=1e-12)
maximum=0
for pa,pb in zip(a.parameters(),b.parameters(),strict=True):
    torch.testing.assert_close(pa.grad,pb.grad,atol=1e-12,rtol=1e-12)
    maximum=max(maximum,float((pa.grad-pb.grad).abs().max()))
proof={'accepted':True,'created':time.time(),'original_samples':len(samples),'merged_segments':len(merged),
       'original_tokens':sum(len(item.tokens) for item in samples),'merged_tokens':sum(len(item.tokens) for item in merged),
       'logical_trajectories':len({item.metadata['polar']['session_id'] for item in merged}),
       'trainable_tokens':sum(map(sum,converted['loss_masks'])),'role_tokens':sum(map(sum,converted['role_masks'])),
       'native_dynamic_batch':manager._dynamic_global_batch_size,'updates_per_batch':1,
       'causal_model_gradient_max_abs':maximum,'causal_model_gradient_tolerance':1e-12,
       'native_dp1_split_accepted':True,'native_advantage_mapping_max_abs':0,
       'actual_qwen_megatron_gradient_equivalence':'pending','production_activation':False,
       'new_model_generation_calls':0}
(PREPARED/'NATIVE_INTEGRATION_CPU_R2_ACCEPTED.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof,indent=2))
