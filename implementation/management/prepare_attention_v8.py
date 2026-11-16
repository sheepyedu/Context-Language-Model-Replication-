"""Freeze the validated dependency recovery without changing scientific inputs."""

import ast
import hashlib
import json
from pathlib import Path
import time

root = Path('/mount-data/CLM-Replication')
accepted = json.loads((root / 'management/FLASH_NATIVE_ROOT_ACCEPTED.json').read_bytes())
assert accepted['accepted'] and accepted['all_D256_gradient_channels_nonzero']
assert json.loads((root / 'management/CACHED_FIRST_BATCH_ROOT_ACCEPTED.json').read_bytes())['accepted']
old = json.loads((root / 'management/SOURCE_PINS.training-v7.json').read_bytes())
files = {}
replacement = {'train-v7.sh': 'train-v8.sh', 'owned_training_v7.py': 'owned_training_v8.py'}
for name, expected in old['files'].items():
    path = Path(name)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    target = Path(name.replace('/source-revisions/training-v7/', '/source-revisions/training-v8/'))
    if target.name in replacement and target.parent == root / 'management':
        target = target.with_name(replacement[target.name])
    files[str(target)] = hashlib.sha256(target.read_bytes()).hexdigest()
for name in ('runtime-attention-v8.sh', 'launch_training_v8.py', 'first_update_acceptance_v2.py'):
    path = root / 'management' / name
    files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
package = root / 'source-revisions/training-v8/clm_replication'
path = package / 'weight_provenance.py'
files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
dependency = root / 'env/attention-fa283'
for path in sorted(dependency.rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
assert any('flash_attn_2_cuda' in name and name.endswith('.so') for name in files)
previous = ast.parse((root / 'source-revisions/training-v7/clm_replication/bridge.py').read_text())
current = ast.parse((package / 'bridge.py').read_text())
for name in ('trajectory_outcome', 'post_process_rewards', 'candidate_decision',
             'generate_rollout', 'collect_candidate_groups'):
    left = next(node for node in previous.body if getattr(node, 'name', None) == name)
    right = next(node for node in current.body if getattr(node, 'name', None) == name)
    assert ast.dump(left) == ast.dump(right)
proof = {'created': time.time(), 'source_revision': 'training-v8', 'files': files,
         'scientific_functions_AST_unchanged': True,
         'scientific_config': 'original train-config-v7.yaml retained, actual source locked by this manifest',
         'original_failure': 'attempt07 exit1, zero optimizer updates',
         'dependency': 'official FlashAttention2.8.3 built against current Torch2.9.1, sm90 only',
         'rejected_cudnn_916_active': False}
path = root / 'management/SOURCE_PINS.training-v8.json'
path.write_text(json.dumps(proof, indent=2))
print(json.dumps({'files': len(files), 'pins_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'scientific_functions_AST_unchanged': True}, indent=2))
