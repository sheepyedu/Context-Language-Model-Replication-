"""Exercise the live-predecessor refusal before any dispatch side effect."""
import json
import fcntl
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import launch_c1

checks = []


def stat_line(pid, start, state='S'):
    fields = [state] + ['0'] * 18 + [str(start)] + ['0'] * 4
    return f'{pid} (owned python) ' + ' '.join(fields)


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    run = root / 'runs/c2-seed1234'
    directory = run / 'attempt-13'
    directory.mkdir(parents=True)
    proc_root = root / 'proc'
    proc = proc_root / '123'
    proc.mkdir(parents=True)
    pointer = {'pid': 123, 'directory': str(directory), 'argv': ['python', 'owner.py'], 'attempt': 13}
    (run / 'CURRENT_OWNER.json').write_text(json.dumps(pointer))
    (directory / 'OWNER_IDENTITY.json').write_text(json.dumps({'pid': 123, 'stat': stat_line(123, 42)}))
    (proc / 'stat').write_text(stat_line(123, 42))
    assert launch_c1.predecessor_status(root, proc_root)['ready'] is False
    checks.append('live_owner_without_exit_refused')
    for original_exit in [0, 1, 134]:
        (directory / 'EXIT.json').write_text(json.dumps({'exit_code': original_exit}))
        try:
            launch_c1.require_closed_predecessor(root, proc_root)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Still-live predecessor admitted')
    checks.append('live_owner_refused_even_with_integer_exit_sidecar')
    (proc / 'stat').write_text(stat_line(123, 42, 'Z'))
    assert not launch_c1.predecessor_status(root, proc_root)['ready']
    checks.append('zombie_original_process_requires_reaping')
    (proc / 'stat').unlink()
    for original_exit in [0, 1, 134]:
        (directory / 'EXIT.json').write_text(json.dumps({'exit_code': original_exit}))
        status = launch_c1.require_closed_predecessor(root, proc_root)
        assert status['original_exit']['exit_code'] == original_exit
    checks.append('gone_owner_original_integer_exit_preserved')
    for value in [None, '0', True]:
        (directory / 'EXIT.json').write_text(json.dumps({'exit_code': value}))
        assert not launch_c1.predecessor_status(root, proc_root)['ready']
    checks.append('unknown_string_and_boolean_exits_refused')
    (directory / 'EXIT.json').unlink()
    assert not launch_c1.predecessor_status(root, proc_root)['ready']
    checks.append('missing_original_exit_refused')
    (directory / 'EXIT.json').write_text(json.dumps({'exit_code': 0}))
    (proc / 'stat').write_text(stat_line(123, 43))
    assert launch_c1.predecessor_status(root, proc_root)['ready']
    checks.append('reused_PID_is_distinguished_by_original_start')
    assert not (root / 'runs/c1-seed1234').exists()
    checks.append('predecessor_inspection_creates_no_C1_run_or_intent')
    c1 = root / 'runs/c1-seed1234'
    c1.mkdir()
    with (c1 / 'DISPATCH.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with patch.object(launch_c1, 'dispatch') as dispatch:
            try:
                launch_c1.admit(SimpleNamespace(attempt=1), {}, root)
            except BlockingIOError:
                pass
            else:
                raise AssertionError('Concurrent admission was allowed')
            dispatch.assert_not_called()
    checks.append('concurrent_attempt_admission_excluded_by_native_file_lock')
    with patch.object(launch_c1, 'dispatch') as dispatch:
        launch_c1.admit(SimpleNamespace(attempt=1), {}, root)
        launch_c1.admit(SimpleNamespace(attempt=2), {}, root)
        assert dispatch.call_count == 2
    checks.append('admission_lock_released_after_return')

proof = {'accepted': True, 'created': time.time(), 'checks': checks, 'count': len(checks),
         'actual_network_model_GPU_or_dispatch_operations': 0,
         'scope': 'predecessor admission predicates only, actual C1 launch pending'}
Path(__file__).with_name('CPU_ACCEPTED.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof, indent=2))
