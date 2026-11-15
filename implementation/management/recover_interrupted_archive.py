"""Finish a known interrupted copy using the unchanged original session tree."""
import hashlib
import json
from pathlib import Path
import shutil
import time

root = Path('/mount-data/CLM-Replication')
sid = 'sk-polar-27f561ae-79f4-4c13-a9be-5b72eeb73cfb'
base = root / 'runs/session-artifacts'
partial = base / (sid + '.partial')
output = base / sid
assert partial.is_dir() and not output.exists()
expected = hashlib.sha256((partial / 'logs/agent/edit_events.json').read_bytes()).hexdigest()
matched = []
for candidate in Path('/tmp').glob('session-sk-polar-*/logs/agent/edit_events.json'):
    if hashlib.sha256(candidate.read_bytes()).hexdigest() == expected:
        matched.append(candidate.parents[2])
assert len(matched) == 1, matched
source = matched[0]
partial_files = [p for p in partial.rglob('*') if p.is_file()]
interrupted_files = []
for path in partial_files:
    target = source / path.relative_to(partial)
    assert target.exists()
    copied = path.read_bytes()
    original = target.read_bytes()
    assert original.startswith(copied), str(path)
    if copied != original:
        interrupted_files.append({'name': str(path.relative_to(partial)),
                                  'copied_bytes': len(copied), 'original_bytes': len(original)})
recovering = base / (sid + '.recovery')
shutil.copytree(source, recovering)
manifest = {}
for path in recovering.rglob('*'):
    if path.is_file():
        name = str(path.relative_to(recovering))
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == value
        manifest[name] = value
assert 'logs/judge/response.raw' in manifest and 'logs/judge/TOKEN_RECEIPT.json' in manifest
(recovering / 'ARCHIVE_MANIFEST.json').write_text(json.dumps(manifest, indent=2))
recovering.rename(output)
proof = {'accepted': True, 'created': time.time(), 'session_id': sid, 'original_source': str(source),
         'interrupted_partial_preserved': str(partial), 'verified_existing_partial_files': len(partial_files),
         'interrupted_copy_files': interrupted_files, 'recovered_files': len(manifest), 'all_original_source_hashes_match': True,
         'model_tool_or_judge_replay': 0,
         'cause': 'Gateway handoff checked terminal session registry before postrun archive worker finished'}
(root / 'management/judge-drain-r1/ARCHIVE_RECOVERY_ACCEPTED.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof, indent=2))
