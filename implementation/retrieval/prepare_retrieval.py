"""Verify the new retrieval assets and install Java inside this project only."""

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


root = Path('/mount-data/CLM-Replication')
assets = root / 'env/retrieval-assets'
proof = json.loads((assets / 'DOWNLOAD_VERIFIED.json').read_text())
index = Path('/mount-data/Datasets/BrowseComp-Plus/indexes/bm25-b3f37f70')
for item in proof['files']:
    path = index / Path(item['name']).name if item['name'].startswith('bm25/') else assets / item['name']
    assert path.stat().st_size == item['size']
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
    assert sha.hexdigest() == item['sha256'], str(path)
java_archive = next(item for item in proof['files'] if item['name'].endswith('.tar.gz'))
java_root = root / 'env/java21'
assert not java_root.exists(), 'Inspect an existing installation before reusing it'
java_root.mkdir()
with tarfile.open(assets / java_archive['name']) as archive:
    archive.extractall(java_root, filter='data')
java_home = next(java_root.iterdir())
version = subprocess.run([str(java_home / 'bin/java'), '-version'], capture_output=True, text=True, check=True)
runtime = root / 'management/retrieval-runtime.sh'
runtime.write_text(
    '#!/bin/bash\nset -euo pipefail\n'
    f'export JAVA_HOME="{java_home}"\n'
    'export PATH="$JAVA_HOME/bin:$PATH"\n'
    'export CUDA_VISIBLE_DEVICES="" USE_TORCH=0 USE_TF=0 HF_HUB_OFFLINE=1\n'
    'export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2\n'
    f'export PYTHONPATH="{root}/management/retrieval-v1:{root}/env/rl/lib/python3.11/site-packages"\n'
    f'exec "{root}/env/retrieval-v1/bin/python" "$@"\n'
)
runtime.chmod(0o755)
accepted = {'assets_verified': len(proof['files']), 'index_revision': proof['index_revision'],
            'java_version': version.stderr.strip(), 'java_home': str(java_home),
            'index_path': str(index), 'runtime': str(runtime), 'cuda_visible_devices': '',
            'active_training_source_changed': False, 'model_calls': 0}
(root / 'management/RETRIEVAL_ASSETS_ACCEPTED.json').write_text(json.dumps(accepted, indent=2))
print(json.dumps(accepted, indent=2))
