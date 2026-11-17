"""Verify a closed retrieval snapshot before publishing immutable records."""

import hashlib
import json
from pathlib import Path
import sys
import tarfile


source = Path(sys.argv[1])
manifest = json.loads((source / 'MANIFEST.json').read_text())
archive_path = source / 'records.tar'
assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == manifest['archive_sha256']
destination = Path('/mount-data/CLM-Replication/runs/web-retrieval')
destination.mkdir(exist_ok=True)
added = 0
with tarfile.open(archive_path) as archive:
    members = archive.getmembers()
    assert {member.name for member in members} == set(manifest['files'])
    for member in members:
        relative = Path(member.name)
        assert member.isfile() and not relative.is_absolute() and '..' not in relative.parts
        content = archive.extractfile(member).read()
        assert hashlib.sha256(content).hexdigest() == manifest['files'][member.name]
        target = destination / relative
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            assert hashlib.sha256(target.read_bytes()).hexdigest() == manifest['files'][member.name]
        else:
            with target.open('xb') as handle:
                handle.write(content)
            added += 1
proof = {
    'accepted': True,
    'closed_retrievals': manifest['closed_retrievals'],
    'verified_files': len(members),
    'new_files': added,
    'original_snapshot': str(source),
    'originals_deleted': False,
    'new_model_tool_or_judge_calls': 0,
}
(source / 'ACCEPTED.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof, indent=2))
