"""Check C1 arguments and native Ray job overrides without creating a GPU actor."""

import json
import os
from pathlib import Path
import time

import ray


ROOT = Path('/mount-data/CLM-Replication')
DIRECTORY = ROOT / 'management/conditions-c1-tp4-v2'


def main():
    base = json.loads((ROOT / 'runs/c2-seed1234/attempt-13/effective_args.json').read_bytes())
    c1 = json.loads((DIRECTORY / 'validation/effective_args.json').read_bytes())
    changed = {key for key in base.keys() | c1.keys() if base.get(key) != c1.get(key)}
    expected = {'custom_config_path', 'train_env_vars', 'save_debug_rollout_data',
                'save', 'clm_run_id', 'save_debug_train_data',
                'clm_admission_gate_path', 'clm_audit_dir', 'polar_task_id_template'}
    assert changed == expected, changed
    assert c1['train_env_vars'] == {**base['train_env_vars'], 'POLAR_DUAL_CHANNEL_W_EFF': '0'}
    assert c1['tensor_model_parallel_size'] == 4 and c1['recompute_num_layers'] == 1
    assert c1['seed'] == c1['rollout_seed'] == 1234
    assert c1['rollout_batch_size'] == 8 and c1['n_samples_per_prompt'] == 32
    assert c1['num_rollout'] == 70
    source = ROOT / 'src/slime/slime/ray/actor_group.py'
    assert '**self.args.train_env_vars,' in source.read_text()
    ray.init(address='127.0.0.1:29477',
             runtime_env={'env_vars': {'POLAR_DUAL_CHANNEL_W_EFF': '0.25'}})

    @ray.remote(num_cpus=0.1, num_gpus=0,
                runtime_env={'env_vars': {**c1['train_env_vars'], 'CUDA_VISIBLE_DEVICES': ''}})
    class EnvironmentProbe:
        def check(self):
            import torch
            from clm_replication.advantages import custom_advantage_function
            from argparse import Namespace

            import importlib.util
            assert importlib.util.find_spec('clm_replication.advantages').origin == '/mount-data/CLM-Replication/source-revisions/training-v10/clm_replication/advantages.py'
            actual = os.environ['POLAR_DUAL_CHANNEL_W_EFF']
            assert actual == '0'
            assert not os.environ.get('CUDA_VISIBLE_DEVICES', '')
            batch = {'kl': [torch.zeros(3)], 'rewards': [1.],
                     'a_eff': [.5], 'role_masks': [[0, 1, 0]]}
            custom_advantage_function(Namespace(), batch)
            assert torch.equal(batch['advantages'][0], torch.ones(3))
            return {'pid': os.getpid(), 'actual_efficiency_weight': actual,
                    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                    'native_task_only_advantages': batch['advantages'][0].tolist(), 'CUDA_initialized': torch.cuda.is_initialized(), 'source_package': '/mount-data/CLM-Replication/source-revisions/training-v10'}

    actor = EnvironmentProbe.remote()
    try:
        result = ray.get(actor.check.remote(), timeout=60)
    finally:
        ray.kill(actor)
        ray.shutdown()
    deadline = time.monotonic() + 15
    while Path('/proc', str(result['pid'])).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('Owned CPU environment-check worker has not exited')
        time.sleep(.2)
    proof = {'accepted': True, 'created': time.time(),
             'scope': 'C1 configuration and CPU-only native Ray environment override',
             'changed_effective_keys': sorted(changed), 'cpu_worker': result,
             'cpu_worker_exactgone': True, 'new_model_calls': 0,
             'gpu_actor_created': False, 'C1_training_started': False,
             'C1_actual_gpu_actor_environment_pending': True}
    (DIRECTORY / 'CPU_ACCEPTED.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
