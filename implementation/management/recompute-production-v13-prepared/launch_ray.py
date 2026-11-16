"""Create the authorized private four-card Ray session after resource release."""

import json
from pathlib import Path
import subprocess
import time

root = Path('/mount-data/CLM-Replication')
directory = root / 'management/recompute-production-v13-prepared'
directory.mkdir(exist_ok=True)
allowed = {
    '4': 'GPU-101b4bf3-3f4e-5706-eef9-7e3d38ba4ea3',
    '5': 'GPU-c64fa322-ea6c-ed73-b81b-67bddfd2ad9e',
    '6': 'GPU-d75359c6-2336-10a8-5eb5-379d8c05d1d0',
    '7': 'GPU-0e09fed4-3dc8-937b-5d89-95ba16b5edbb',
}
applications = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                       '--format=csv,noheader'], text=True)
assert not any(uuid in applications for uuid in allowed.values())
argv = [str(root / 'management/runtime-gpu4567.sh'), 'ray', 'start', '--head',
        '--port=29477', '--node-ip-address=127.0.0.1', '--num-cpus=24', '--num-gpus=4',
        '--object-store-memory=17179869184', '--dashboard-host=127.0.0.1',
        '--dashboard-port=29478', '--ray-client-server-port=29479',
        '--min-worker-port=29500', '--max-worker-port=29627',
        '--dashboard-agent-listen-port=29480', '--dashboard-agent-grpc-port=29481',
        '--metrics-export-port=29482', '--runtime-env-agent-port=29483',
        '--node-manager-port=29484', '--object-manager-port=29485',
        '--temp-dir=/tmp/clm-ray-unmerged-v13-20261116', '--disable-usage-stats']
with (directory / 'RAY_INTENT.json').open('x') as handle:
    json.dump({'created': time.time(), 'argv': argv, 'physical_gpus': allowed}, handle, indent=2)
with (directory / 'RAY_START.log').open('wb') as log:
    child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(directory / 'RAY_LAUNCH.json').write_text(json.dumps({'pid': child.pid, 'created': time.time()}))
print(json.dumps({'pid': child.pid, 'physical_gpus': list(allowed)}))
