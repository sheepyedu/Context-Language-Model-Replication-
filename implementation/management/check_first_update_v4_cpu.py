"""Exercise the actual admission/parser helpers without HTTP, CUDA or a model."""
import importlib.util
import json
from pathlib import Path
import tempfile

path = Path(__file__).with_name('first_update_acceptance_v4.py')
spec = importlib.util.spec_from_file_location('first_update_v4', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
checks = []
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    proc = root / '123'
    proc.mkdir()
    # Attempt 13 deliberately uses owner implementation v14.
    launch = {'pid': 123, 'attempt': 13,
              'argv': ['/env/python', '/management/owned_training_v14.py', '--attempt', '13']}
    owner = {**launch, 'directory': '/runs/attempt-13', 'created': 0}
    fields = ['S', '1'] + ['0'] * 17 + ['51919204'] + ['0'] * 10
    (proc / 'stat').write_text('123 (python worker) ' + ' '.join(fields))
    raw = b'\0'.join(value.encode() for value in launch['argv']) + b'\0'
    (proc / 'cmdline').write_bytes(raw)
    result = module.verify_owner_identity(owner, launch, 51919204, root)
    assert result['start_ticks'] == 51919204
    checks.append('attempt13_owner_v14_exact_argv_accepted')

    def rejected(call, label):
        try:
            call()
        except ValueError:
            checks.append(label)
        else:
            raise AssertionError(label)

    rejected(lambda: module.verify_owner_identity(owner, launch, 1, root), 'reused_pid_start_rejected')
    rejected(lambda: module.verify_owner_identity({**owner, 'attempt': 14}, launch, 51919204, root),
             'pointer_launch_attempt_mismatch_rejected')
    (proc / 'cmdline').write_bytes(raw.replace(b'v14.py', b'v13.py'))
    rejected(lambda: module.verify_owner_identity(owner, launch, 51919204, root),
             'incorrect_source_version_rejected')
    (proc / 'cmdline').write_bytes(raw + b'extra\0')
    rejected(lambda: module.verify_owner_identity(owner, launch, 51919204, root), 'extra_argv_rejected')
    (proc / 'cmdline').write_bytes(raw)
    fields[0] = 'Z'
    (proc / 'stat').write_text('123 (python worker) ' + ' '.join(fields))
    rejected(lambda: module.verify_owner_identity(owner, launch, 51919204, root), 'zombie_rejected')
    log = root / 'controller.log'
    log.write_text("step 0: {'train/grad_norm': 1.2, 'train/step': 1, 'train/pg_loss': -0.1}\n")
    assert len(module.native_actor_metrics(log)) == 1
    checks.append('native_update_metric_parsed')
    changes = module.changed_language_tensors({'success': False, 'message':
        'check tensor equality failed: name=model.language_model.layers.0.weight max_abs_err=0.0001 mean_abs_err=0.00001'})
    assert len(changes) == 1
    checks.append('finite_positive_language_difference_parsed')
print(json.dumps({'accepted': True, 'checks': checks, 'total': len(checks),
                  'actual_model_optimizer_updates': 0, 'network_calls': 0}, indent=2))
