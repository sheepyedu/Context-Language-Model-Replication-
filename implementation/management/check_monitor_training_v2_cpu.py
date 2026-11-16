"""Regress optimizer evidence aging out of the compact progress tail."""

import json
from pathlib import Path
import tempfile

from monitor_training_current_v2 import scan_optimizer_metrics


checks = []
with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / 'controller.log'
    metric = 'native train/step=0 train/grad_norm=0.125 train/loss=0.25\n'
    suffix = b'actor train: still computing\n' * 8000
    path.write_bytes(metric.encode() + suffix)
    old_tail = path.read_bytes()[-131072:].decode()
    old_rows = [line for line in old_tail.splitlines()
                if 'train/grad_norm' in line and 'train/step' in line]
    assert old_rows == []
    rows, proof = scan_optimizer_metrics(path)
    assert proof['complete'] and rows == [metric.strip()]
    checks.append('RED_old_128KiB_tail_loses_metric_GREEN_bounded_full_scan_preserves_it')

    path.write_text(''.join(f'native train/step={i} train/grad_norm=0.125\n' for i in range(5)))
    rows, proof = scan_optimizer_metrics(path)
    assert len(rows) == 3 and 'train/step=2 ' in rows[0]
    assert proof['matching_lines'] == 5
    checks.append('output_remains_last_three_rows_while_coverage_scans_all_original_bytes')

    path.write_bytes(b'prefix train/step=0\ntrain/grad_norm=0.1\n')
    rows, proof = scan_optimizer_metrics(path)
    assert rows == [] and proof['complete']
    checks.append('both_native_metric_keys_must_be_in_the_same_original_line')

    path.write_bytes(b'')
    rows, proof = scan_optimizer_metrics(path)
    assert rows == [] and proof['complete'] and proof['snapshot_bytes'] == 0
    checks.append('empty_native_log_has_explicit_zero_byte_coverage')

    path.write_bytes(metric.encode())
    ticks = iter([0, 1])
    rows, proof = scan_optimizer_metrics(path, max_seconds=1, clock=lambda: next(ticks))
    assert not proof['complete'] and proof['scanned_bytes'] == 0 and rows == []
    checks.append('time_budget_expiry_is_incomplete_coverage_not_proof_of_zero_updates')

    path.write_bytes(metric.encode())
    ticks = [0]

    def append_once():
        ticks[0] += 1
        if ticks[0] == 2:
            with path.open('ab') as handle:
                handle.write(b'native train/step=1 train/grad_norm=0.2\n')
        return 0

    rows, proof = scan_optimizer_metrics(path, clock=append_once)
    assert proof['complete'] and proof['snapshot_bytes'] == len(metric.encode())
    assert rows == [metric.strip()]
    checks.append('concurrent_appends_do_not_extend_the_frozen_scan_boundary')

    path.write_bytes(metric.encode())
    ticks = [0]

    def truncate_once():
        ticks[0] += 1
        if ticks[0] == 2:
            path.write_bytes(b'')
        return 0

    rows, proof = scan_optimizer_metrics(path, clock=truncate_once)
    assert not proof['complete'] and rows == []
    checks.append('truncated_log_is_incomplete_coverage')

    path.write_bytes(b'\xff native train/step=0 train/grad_norm=0.1\n')
    rows, proof = scan_optimizer_metrics(path)
    assert proof['complete'] and rows[0].startswith('\ufffd')
    checks.append('raw_invalid_UTF8_has_visible_replacement_without_crashing_scan')

print(json.dumps({'accepted': True, 'checks': checks, 'count': len(checks),
                  'actual_network_GPU_or_model_calls': 0}, indent=2))
