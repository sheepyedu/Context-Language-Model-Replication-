"""Reproduce copied Ray args and compare the old and new resume boundary."""

import ast
import json
import os
from pathlib import Path
import time
from types import ModuleType, SimpleNamespace
import uuid

import ray


def run_boundary(source, args):
    import tempfile
    from unittest.mock import patch

    tree = ast.parse(source)
    update = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'update_policy_version')
    generate = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'generate_rollout')
    guard = next(node for node in generate.body
                 if isinstance(node, ast.If) and ast.unparse(node.test).startswith('_served_policy_version !='))
    resume = next(node for node in generate.body
                  if isinstance(node, ast.If) and 'CLM_RESUME_INITIAL_GATEWAY' in ast.unparse(node.test))
    hooks, cursors, http = [], [], []
    parent = ModuleType('clm_replication')
    parent.__path__ = []
    cursor = ModuleType('clm_replication.checkpoint_cursor')
    cursor.record_loaded_cursor = lambda *values: cursors.append(values[1])
    upstream = ModuleType('slime_bridge')
    upstream.__path__ = []
    delegate = ModuleType('slime_bridge.rollout')
    delegate.update_policy_version = lambda args, version: hooks.append(version)

    def post(url, timeout):
        http.append(url)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'paused': False})

    with tempfile.TemporaryDirectory() as temporary:
        args.clm_audit_dir = temporary
        args.polar_gateway_url = 'http://invalid.test'
        namespace = {'__package__': 'clm_replication', 'args': args, 'data_source': object(),
                     '_served_policy_version': None, '_initial_synced_policy_version': None,
                     'rollout_id': 5, 'Path': Path, 'json': json, 'os': os}
        modules = {'clm_replication': parent, 'clm_replication.checkpoint_cursor': cursor,
                   'slime_bridge': upstream, 'slime_bridge.rollout': delegate,
                   'httpx': SimpleNamespace(post=post)}
        with patch.dict('sys.modules', modules), patch.dict(os.environ, {'CLM_RESUME_INITIAL_GATEWAY': '1'}):
            exec(compile(ast.fix_missing_locations(ast.Module(body=[update], type_ignores=[])),
                         'original-update-hook', 'exec'), namespace)
            namespace['update_policy_version'](args, 5)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[guard, resume], type_ignores=[])),
                         'original-generate-prefix', 'exec'), namespace)
            namespace['update_policy_version'](args, 6)
            namespace['update_policy_version'](args, 7)
        return {'HTTP_resume_calls': len(http), 'cursor_checks': cursors, 'hook_versions': hooks,
                'initial_synced_policy': namespace['_initial_synced_policy_version'],
                'latest_synced_policy': namespace['_served_policy_version']}


@ray.remote(num_cpus=0.1, num_gpus=0)
class BoundaryProbe:
    def __init__(self, args):
        self.args = args

    def copied_start(self):
        return self.args.start_rollout_id

    def check(self, old_source, new_source):
        assert self.args.start_rollout_id is None
        old = run_boundary(old_source, self.args)
        new = run_boundary(new_source, self.args)
        assert old['HTTP_resume_calls'] == 0
        assert new['HTTP_resume_calls'] == 1 and new['cursor_checks'] == [5]
        assert new['initial_synced_policy'] == 5 and new['latest_synced_policy'] == 7
        assert new['hook_versions'] == [5, 6, 7]
        return {'old': old, 'new': new, 'pid': os.getpid(),
                'start_ticks': int(Path('/proc/self/stat').read_text().split()[21]),
                'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'assigned_GPU_ids': ray.get_gpu_ids()}


if __name__ == '__main__':
    directory = Path(__file__).parent
    old = (directory / 'BASE_PREPARED_V2.py').read_text()
    new = (directory / 'bridge.py').read_text()
    ray.init(address='127.0.0.1:29477', namespace='clm-cursor-cpu-' + uuid.uuid4().hex,
             runtime_env={'env_vars': {'CUDA_VISIBLE_DEVICES': ''}})
    args = SimpleNamespace(start_rollout_id=None)
    actor = BoundaryProbe.remote(args)
    try:
        assert ray.get(actor.copied_start.remote(), timeout=10) is None
        args.start_rollout_id = 5
        value = ray.get(actor.check.remote(old, new), timeout=15)
        assert value['CUDA_VISIBLE_DEVICES'] == '' and value['assigned_GPU_ids'] == []
    finally:
        ray.kill(actor, no_restart=True)
    deadline = time.monotonic() + 6
    while Path('/proc', str(value['pid'])).exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not Path('/proc', str(value['pid'])).exists()
    ray.shutdown()
    proof = {'accepted': True, 'created': time.time(), 'actual_CPU_Ray_copy_and_hook': value,
             'actual_CPU_actor_exactgone': True, 'native_model_or_optimizer_loaded': False,
             'new_HTTP_or_model_requests': 0}
    (directory / 'RAY_CPU_ACCEPTED.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof))
