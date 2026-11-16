"""Prepare a separate gateway package with driver-owned corpus routing."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time
R=Path('/mount-data/CLM-Replication')
P=R/'management/retrieval-gateway-prepared-v2'
base=R/'source-revisions/training-v10/clm_replication'
target=P/'source/clm_replication'
assert not target.exists(), 'Prepared source already exists; inspect its provenance before changing it'
shutil.copytree(base,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
original=(base/'gateway.py').read_text()
assert hashlib.sha256(original.encode()).hexdigest()=='1314e11a86e5ec02eae0b1b3ae757b54e015bcf30d6e5d2056f882c229613f91'
old="""    async with httpx.AsyncClient(timeout=45, trust_env=False) as client:
        response = await client.post('http://127.0.0.1:19180', json=payload)
        response.raise_for_status()
        return response.json()
"""
new="""    info = get_state().session_registry.get(session_id)
    if info is None:
        raise HTTPException(status_code=403, detail='An active task session is required')
    try:
        backend = retrieval_backend(info.metadata)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    async with httpx.AsyncClient(timeout=45, trust_env=False) as client:
        if backend == BACKENDS['public_web']:
            response = await client.post(backend, json=payload)
            response.raise_for_status()
            return response.json()
        try:
            return await retrieve(payload, info.metadata, client)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except httpx.HTTPStatusError as error:
            raise HTTPException(status_code=error.response.status_code,
                                detail=error.response.json()) from error
"""
assert original.count(old)==1
updated=original.replace(old,new).replace('from .accounting import record_tokens','from .accounting import record_tokens\nfrom .retrieval_routing import BACKENDS, retrieval_backend, retrieve')
(target/'gateway.py').write_text(updated)
shutil.copy2(P/'retrieval_routing.py',target/'retrieval_routing.py')
def functions(text):
 return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
a,b=functions(original),functions(updated)
assert a.keys()==b.keys()
assert all(a[k]==b[k] for k in a if k!='public_web')
assert updated.replace(new,old).replace('\nfrom .retrieval_routing import BACKENDS, retrieval_backend, retrieve','')==original
for f in base.glob('*.py'):
 if f.name!='gateway.py':assert f.read_bytes()==(target/f.name).read_bytes()
for f in target.glob('*.py'):ast.parse(f.read_text())
proof={'accepted':True,'created':time.time(),'base_gateway_sha256':hashlib.sha256(original.encode()).hexdigest(),'candidate_gateway_sha256':hashlib.sha256(updated.encode()).hexdigest(),'only_changed_native_function':'public_web','other_package_files_equal':True,'source':'source-revisions/training-v10','package':str(target),'active_production_gateway_changed':False,'model_calls':0}
(P/'SOURCE_CPU_ACCEPTED.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
