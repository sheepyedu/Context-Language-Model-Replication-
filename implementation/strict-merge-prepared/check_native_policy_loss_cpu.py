"""Exercise actual native PPO/TIS/KL reducers on a causal CPU fixture."""
import ast
from argparse import Namespace
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch
import torch
import torch.nn.functional as F

ROOT=Path('/mount-data/CLM-Replication')
PREPARED=ROOT/'management/strict-merge-candidate-v2'
sys.path[:0]=[str(ROOT/'source-revisions/training-v10'),str(ROOT/'src'),str(ROOT/'src/slime'),str(ROOT/'src/Megatron-LM')]
from slime.utils.types import Sample
from slime.backends.megatron_utils import loss as native
for name in ['strict_merge','strict_training']:
    spec=importlib.util.spec_from_file_location('clm_replication.'+name,PREPARED/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
from clm_replication.strict_training import merge_prepared_samples
# Reuse the fixture declarations, without executing or rewriting the previous audit.
previous=ast.parse((PREPARED/'check_native_integration_r2.py').read_text())
selected=[node for node in previous.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in ['CausalModel','synthetic']]
exec(compile(ast.Module(body=selected,type_ignores=[]),'<existing causal fixture>','exec'),globals())
assert not torch.cuda.is_available()
torch.set_num_threads(4)
fixture=[synthetic([1,2,3,4,5],2,0),synthetic([1,2,3,4,5,20,21,6,7],2,1),
         synthetic([1,2,3,4,5,20,21,6,7,22,8,9],2,2)]
joined,_=merge_prepared_samples(fixture,[1]*3,[.8]*3)
assert len(joined)==1
args=Namespace(**json.loads((ROOT/'runs/c2-seed1234/attempt-11/effective_args.json').read_text()))

def cpu_logprobs(logits,*,args,unconcat_tokens,total_lengths,response_lengths,with_entropy=False,**kwargs):
    values=[];entropies=[]
    offset=0
    for tokens,length,response in zip(unconcat_tokens,total_lengths,response_lengths,strict=True):
        logp=(logits[0,offset+length-response-1:offset+length-1]/args.rollout_temperature).log_softmax(-1)
        values.append(logp.gather(-1,tokens[-response:,None]).squeeze(-1))
        entropies.append(-(logp.exp()*logp).sum(-1))
        offset+=length
    return logits.new_empty(0),{'log_probs':values,'entropy':entropies}

def objective(model,items,old_offset,reference_offset):
    denominator=sum(sum(item.loss_mask) for item in items)
    objective=torch.zeros((),dtype=torch.float64);values=[];normalizers=[]
    for item in items:
        logits=model(item.tokens).float()[None]
        base={'unconcat_tokens':[torch.tensor(item.tokens)],'total_lengths':[len(item.tokens)],
              'response_lengths':[item.response_length],
              'loss_masks':[torch.tensor(item.loss_mask,dtype=torch.float32)]}
        _,terms=cpu_logprobs(logits,args=args,**base)
        current=terms['log_probs'][0]
        base.update(advantages=[.8+.25*.15*torch.tensor(item.metadata['polar']['role_mask'])],
                    log_probs=[current.detach()+old_offset],
                    ref_log_probs=[current.detach()+reference_offset],
                    rollout_log_probs=[torch.tensor(item.rollout_log_probs)])
        loss,normalizer,_=native.loss_function(args,base,len(items),logits)
        objective=objective+loss/denominator
        normalizers.append(float(normalizer))
        values.extend(current[base['loss_masks'][0].bool()].detach().tolist())
    return objective,torch.tensor(values),sum(normalizers)

tests=[]
with patch.object(native.mpu,'get_context_parallel_world_size',return_value=1), \
     patch.object(native,'get_log_probs_and_entropy',cpu_logprobs):
    for old_offset,reference_offset in [(.015,-.012),(.7,.2),(-.7,-.2),(0,0)]:
        torch.manual_seed(20261116)
        a=CausalModel();b=deepcopy(a)
        loss_a,values_a,count_a=objective(a,fixture,old_offset,reference_offset)
        loss_b,values_b,count_b=objective(b,joined,old_offset,reference_offset)
        torch.testing.assert_close(values_a,values_b,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(loss_a,loss_b,atol=1e-6,rtol=1e-6)
        assert count_a==count_b==6
        loss_a.backward();loss_b.backward()
        maximum=0.
        for pa,pb in zip(a.parameters(),b.parameters(),strict=True):
            torch.testing.assert_close(pa.grad,pb.grad,atol=1e-6,rtol=1e-6)
            maximum=max(maximum,float((pa.grad-pb.grad).abs().max()))
        tests.append({'old_offset':old_offset,'reference_offset':reference_offset,
                      'gradient_max_abs':maximum,'loss_abs_diff':abs(float(loss_a-loss_b))})
proof={'accepted':True,'cases':tests,'tolerance_abs_and_relative':1e-6,
       'actual_native_loss_function':True,'actual_native_policy_loss_TIS_KL_reducers':True,
       'vocabulary_logprobs':'CPU reference replacement, not GPU vocabulary kernel',
       'model':'small causal attention fixture, not full Qwen/Megatron',
       'production_activated':False,'new_model_generation_calls':0,'actual_optimizer_updates':0,
       'full_Qwen_Megatron_gradient_equivalence':'pending'}
target=PREPARED/'NATIVE_POLICY_LOSS_CPU_ACCEPTED.json'
with target.open('x') as handle:json.dump(proof,handle,indent=2)
print(json.dumps(proof,indent=2))
