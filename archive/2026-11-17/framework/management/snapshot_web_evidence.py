"""Snapshot only closed public-retrieval records for durable CPFS retention."""

import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time


root = Path(__file__).resolve().parent
records = root / 'web-retrieval'
output = root / 'web-evidence-snapshots' / str(time.time_ns())
output.mkdir(parents=True)
manifest = {}
with tarfile.open(output / 'records.tar', 'w') as archive:
    for receipt_path in sorted(records.glob('*/receipt.json')):
        try:
            receipt_bytes = receipt_path.read_bytes()
            receipt = json.loads(receipt_bytes)
        except json.JSONDecodeError:
            # The broker may still be writing this last record.
            continue
        body_path = receipt_path.with_name('body.raw.gz')
        body_bytes = body_path.read_bytes()
        raw = gzip.decompress(body_bytes)
        if hashlib.sha256(raw).hexdigest() != receipt['body_sha256']:
            raise ValueError(f'Retrieval body mismatch: {body_path}')
        for path, content in ((receipt_path, receipt_bytes), (body_path, body_bytes)):
            name = str(path.relative_to(records))
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(content))
            manifest[name] = hashlib.sha256(content).hexdigest()
proof = {
    'created': time.time(),
    'files': manifest,
    'closed_retrievals': len(manifest) // 2,
    'archive_sha256': hashlib.sha256((output / 'records.tar').read_bytes()).hexdigest(),
    'new_model_tool_or_judge_calls': 0,
}
(output / 'MANIFEST.json').write_text(json.dumps(proof, indent=2))
print(output)
