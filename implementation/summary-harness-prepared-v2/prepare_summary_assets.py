"""Build isolated tokenizer dependencies and a separate Summary task bundle."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sysconfig
import time

root = Path('/mount-data/CLM-Replication')
prepared = root/'management/summary-harness-prepared-v2'
site = Path(sysconfig.get_paths()['purelib'])
deps = root/'env/summary-agent-deps'
deps.mkdir(exist_ok=False)
proof = {'created':time.time(),'packages':{},'tokenizer_files':{},'new_model_calls':0,'active_C2_source_changed':False}
for name in ['transformers','tokenizers','safetensors']:
    distribution = importlib.metadata.distribution(name)
    info = Path(distribution._path)
    package = site/name
    binaries = list(package.rglob('*.so'))
    assert all(path.name.endswith('.abi3.so') for path in binaries),name
    shutil.copytree(package,deps/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(info,deps/info.name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    proof['packages'][name]={'version':distribution.version,'source':str(package),
                             'binary_abi':'abi3' if binaries else 'pure_python'}
assets = root/'summary-assets/tokenizer'
assets.mkdir(parents=True,exist_ok=False)
for name in ['config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt',
             'special_tokens_map.json','added_tokens.json','chat_template.jinja']:
    source = root/'../Models/Qwen3.5-9B'/name
    if source.exists():
        destination = assets/name
        shutil.copy2(source,destination)
        assert hashlib.sha256(source.read_bytes()).hexdigest()==hashlib.sha256(destination.read_bytes()).hexdigest()
        proof['tokenizer_files'][name]={'bytes':destination.stat().st_size,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
assert 'tokenizer.json' in proof['tokenizer_files'] and 'tokenizer_config.json' in proof['tokenizer_files']
bundle = root/'agent-bundle-summary-v1/clm_replication'
shutil.copytree(root/'agent-bundle-v1/clm_replication',bundle,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['summary_runner.py','summary_loop.py','summary_polar_agent.py','summary_context.py']:
    shutil.copy2(prepared/name,bundle/name)
shutil.copytree(prepared/'prompts',bundle/'summary_prompts')
for path in sorted(deps.rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts:
        proof.setdefault('dependency_sha256',{})[str(path.relative_to(deps))]=hashlib.sha256(path.read_bytes()).hexdigest()
for path in sorted(bundle.rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts:
        proof.setdefault('bundle_sha256',{})[str(path.relative_to(bundle))]=hashlib.sha256(path.read_bytes()).hexdigest()
proof['prepared']=True
(prepared/'ASSETS_PREPARED.json').write_text(json.dumps(proof,indent=2))
print(json.dumps({'prepared':True,'packages':proof['packages'],'bundle_files':len(proof['bundle_sha256']),
                  'dependency_files':len(proof['dependency_sha256']),'tokenizer_files':len(proof['tokenizer_files']),
                  'actual_agent_import':'pending','S0_S1_started':False},indent=2))
