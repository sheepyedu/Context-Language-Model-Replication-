import json
from pathlib import Path
import subprocess
import time
import argparse

root = Path('/mount-data/CLM-Replication')
parser = argparse.ArgumentParser()
parser.add_argument('--attempt', type=int, required=True)
attempt = parser.parse_args().attempt
with (root / f'management/conversion-{attempt}.INTENT.json').open('x') as handle:
    json.dump({'script': str(root / 'management/convert.sh'), 'created': time.time()}, handle)
with (root / f'management/conversion-{attempt}.log').open('wb') as log:
    process = subprocess.Popen(['bash', str(root / 'management/convert.sh'), str(attempt)],
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
receipt = {'pid': process.pid, 'created': time.time()}
(root / f'management/conversion-{attempt}.launch.json').write_text(json.dumps(receipt))
print(receipt)
