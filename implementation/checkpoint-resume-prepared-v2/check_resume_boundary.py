"""Exercise actual resume-boundary AST without HTTP, imports or model calls."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

root=Path(__file__).resolve().parent
source=(root/'bridge.py').read_text()
tree=ast.parse(source)
function=next(item for item in tree.body if isinstance(item,ast.FunctionDef) and item.name=='generate_rollout')
policy_guard=next(item for item in function.body if isinstance(item,ast.If) and isinstance(item.test,ast.Compare)
                  and isinstance(item.test.left,ast.Name) and item.test.left.id=='_served_policy_version')
resume=next(item for item in function.body if isinstance(item,ast.If) and 'CLM_RESUME_INITIAL_GATEWAY' in ast.unparse(item.test))
body=ast.Module(body=[policy_guard, resume],type_ignores=[])
code=compile(ast.fix_missing_locations(body),str(root/'bridge.py'),'exec')
original_httpx=sys.modules.get('httpx')
old_env=os.environ.get('CLM_RESUME_INITIAL_GATEWAY')
checks=0
try:
 for start in [0,5,70]:
  for rollout in [start,max(0,start-1),start+1]:
   for enabled in [False,True]:
    for actual_policy in [rollout,rollout+1]:
     calls=[]
     def post(url,timeout):
      calls.append((url,timeout))
      return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'paused':False})
     sys.modules['httpx']=SimpleNamespace(post=post)
     os.environ['CLM_RESUME_INITIAL_GATEWAY']='1' if enabled else '0'
     with tempfile.TemporaryDirectory() as directory:
      namespace={'_served_policy_version':actual_policy,'rollout_id':rollout,'Path':Path,'json':json,'os':os,
                 'args':SimpleNamespace(start_rollout_id=start,clm_audit_dir=directory,polar_gateway_url='http://invalid.test')}
      rejected=False
      try:exec(code,namespace)
      except ValueError:rejected=True
      expected=enabled and rollout==start and actual_policy==rollout
      assert len(calls)==int(expected)
      assert rejected==(actual_policy!=rollout)
      receipts=list(Path(directory).glob('*.json'))
      assert len(receipts)==int(expected)
      if expected:
       assert json.loads(receipts[0].read_text())['policy_version']==rollout
       assert f'{rollout:05d}-{os.getpid()}' in receipts[0].name
      checks+=1
finally:
 if original_httpx is None:sys.modules.pop('httpx',None)
 else:sys.modules['httpx']=original_httpx
 if old_env is None:os.environ.pop('CLM_RESUME_INITIAL_GATEWAY',None)
 else:os.environ['CLM_RESUME_INITIAL_GATEWAY']=old_env
proof={'accepted':True,'checks':checks,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
       'sync_guard_before_resume':True,'policy_specific_receipt':True,'new_HTTP_requests':0,'new_model_calls':0,
       'active_source_changed':False,'production_checkpoint_restore':'pending'}
(root/'CPU_ACCEPTED.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof,indent=2))
