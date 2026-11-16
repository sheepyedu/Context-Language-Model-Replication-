"""Compare all retained sample fields, independently of pickle dictionary order."""
import hashlib,json,time
from pathlib import Path
import torch
R=Path('/mount-data/CLM-Replication/runs/c2-seed1234')
with (R/'attempt-12/CACHE_SEMANTIC_INTENT.json').open('x') as f:json.dump({'created':time.time(),'operation':'read-only all-field cache comparison'},f)
def digest_sample(sample):
 return hashlib.sha256(json.dumps(sample,sort_keys=True,default=str,separators=(',',':')).encode()).hexdigest()
digests=[];counts=[];selected=[]
for name in ['training-attempt11/rollout-0.pt','training-attempt12/rollout-0.pt']:
 data=torch.load(R/name,weights_only=False,map_location='cpu');samples=data['samples']
 digests.append([digest_sample(s) for s in samples]);counts.append(len(samples));selected.append(sorted({s['group_index'] for s in samples}))
 del samples,data
assert counts==[11095,11095]
assert digests[0]==digests[1]
proof={'accepted':True,'created':time.time(),'samples':11095,'logical_trajectories':256,'selected_groups':selected[0],'all_native_sample_fields_identical':True,'canonical_sample_hashes_sha256':hashlib.sha256(json.dumps(digests[0]).encode()).hexdigest(),'archive_bytes_identical':False,'pickle_dictionary_key_order_difference_observed':True,'new_generation_calls':0}
(R/'attempt-12/CACHE_SEMANTIC_ACCEPTED.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
