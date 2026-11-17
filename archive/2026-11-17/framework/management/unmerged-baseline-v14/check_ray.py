"""Check Ray's physical GPU assignment without initializing CUDA."""

import json
import os
from pathlib import Path
import time

import ray

ray.init(address='127.0.0.1:29477',runtime_env={'env_vars':{'PYTHONPATH':'/mount-data/CLM-Replication/source-revisions/training-v10:/mount-data/CLM-Replication/src:/mount-data/CLM-Replication/src/ProRL/src:/mount-data/CLM-Replication/src/slime:/mount-data/CLM-Replication/src/Megatron-LM'}})


@ray.remote(num_gpus=1, num_cpus=1)
class Mapping:
    def identity(self):
        import importlib.util
        assert importlib.util.find_spec('clm_replication.bridge').origin == '/mount-data/CLM-Replication/source-revisions/training-v10/clm_replication/bridge.py'
        return {'pid': os.getpid(), 'ids': ray.get_gpu_ids(),
                'visible': os.environ.get('CUDA_VISIBLE_DEVICES')}


workers = [Mapping.remote() for _ in range(4)]
try:
    mappings = ray.get([worker.identity.remote() for worker in workers], timeout=60)
    assert sorted(row['visible'] for row in mappings) == ['4', '5', '6', '7']
    resources = ray.cluster_resources()
    assert resources['GPU'] == 4
    proof = {'accepted': True, 'created': time.time(), 'resources': resources,
             'GPU_mappings': mappings,
             'session_path': str(Path('/tmp/clm-ray-unmerged-v13-20261116/session_latest').resolve()),
             'CUDA_initialized': False, 'new_generation_calls': 0}
    Path('/mount-data/CLM-Replication/management/unmerged-baseline-v14/RAY_ACCEPTED.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof, indent=2))
finally:
    for worker in workers:
        ray.kill(worker, no_restart=True)
ray.shutdown()
