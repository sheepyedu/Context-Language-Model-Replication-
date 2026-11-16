"""Derive C1's owned controller from the accepted C2 controller."""
import ast
import hashlib
import json
from pathlib import Path
import time

ROOT = Path('/mount-data/CLM-Replication')
PREPARED = ROOT / 'management/c1-owned-entry-prepared-v1'
base = (ROOT / 'management/owned_training_v14.py').read_text()
text = base.replace('runs/c2-seed1234/attempt-', 'runs/c1-seed1234/attempt-')
text = text.replace("root / 'management/train-v14.sh'",
                    "root / 'management/conditions-c1-tp4-v2/train-c1-tp4.sh'")
assert text != base
assert text.replace('runs/c1-seed1234/attempt-', 'runs/c2-seed1234/attempt-').replace(
    "root / 'management/conditions-c1-tp4-v2/train-c1-tp4.sh'",
    "root / 'management/train-v14.sh'") == base
ast.parse(text)
target = PREPARED / 'owned_c1.py'
with target.open('x') as handle:
    handle.write(text)
proof = {'accepted': True, 'created': time.time(), 'prepared_only': True,
         'base_owner_sha256': hashlib.sha256(base.encode()).hexdigest(),
         'candidate_owner_sha256': hashlib.sha256(text.encode()).hexdigest(),
         'changes': ['C1 run directory', 'accepted C1 task-only launch script'],
         'original_router_wait_and_exit_collection_unchanged': True,
         'new_model_or_judge_calls': 0}
(PREPARED / 'OWNER_DERIVATION_ACCEPTED.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof, indent=2))
