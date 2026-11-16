"""Check bounded GET export and the public reporting CLI with fake responses."""
import fcntl
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.error

from collect_task_results import collect

HERE = Path(__file__).resolve().parent
checks = []


class Response:
    status = 200
    def __init__(self, raw):
        self.stream = io.BytesIO(raw)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self, size):
        return self.stream.read(size)


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    manifest = {'condition': 'fixture', 'purpose': 'validation', 'checkpoint_step': 5,
                'dataset_sha256': 'fixture-data', 'terminal_session_statuses': ['COMPLETED', 'ERROR', 'TIMEOUT'],
                'cases': [{'qid': 'A', 'task_id': 'task A/中文'}]}
    manifest_raw = json.dumps(manifest).encode()
    original = {'task_id': 'task A/中文', 'status': 'completed', 'results': [{
        'task_id': 'task A/中文', 'session_id': 'fixture-sid', 'status': 'COMPLETED',
        'trajectory': {'traces': [], 'metadata': {'policy_version': 5,
            'task_metadata': {'qid': 'A', 'condition': 'fixture', 'purpose': 'validation'},
            'evaluation': {'outcome_reward': 1.0, 'clm': {
                'success': True, 'judge': {'success': True}, 'flops': 100}}}}}]}
    original_raw = json.dumps(original, ensure_ascii=False).encode()
    calls = []
    def opener(request, timeout):
        assert request.method == 'GET'
        assert request.full_url.endswith('task%20A%2F%E4%B8%AD%E6%96%87')
        calls.append(request.full_url)
        return Response(original_raw)
    saved = collect(manifest_raw, root / 'capture', 'http://cpu-fixture.invalid', opener=opener)
    assert saved['terminal_task_responses'] == 1 and len(calls) == 1
    index = json.loads(Path(saved['immutable_index']).read_text())
    record = index['responses']['task A/中文']
    assert Path(record['path']).read_bytes() == original_raw
    assert record['sha256'] == hashlib.sha256(original_raw).hexdigest()
    checks.append('GET_only_export_preserves_original_bytes_hash_and_encoded_task_ID')
    again = collect(manifest_raw, root / 'capture', 'http://cpu-fixture.invalid', opener=opener)
    assert again['coordinator_GETs_this_batch'] == 0 and len(calls) == 1
    assert again['immutable_index'] == saved['immutable_index']
    checks.append('closed_task_and_identical_index_reused_without_another_GET')
    changed = {**manifest, 'checkpoint_step': 10}
    try:
        collect(json.dumps(changed).encode(), root / 'capture', 'http://cpu-fixture.invalid', opener=opener)
    except ValueError:
        pass
    else:
        raise AssertionError('Different manifest was mixed into the same capture')
    checks.append('capture_directory_bound_to_exact_manifest_SHA')
    with (root / 'capture/EXPORT.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            collect(manifest_raw, root / 'capture', 'http://cpu-fixture.invalid', opener=opener)
        except BlockingIOError:
            pass
        else:
            raise AssertionError('Concurrent collector was admitted')
    checks.append('concurrent_one_shot_export_excluded_by_native_lock')
    def rejected(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, 'fixture', {}, io.BytesIO(b'{"detail":"missing"}'))
    failed = collect(manifest_raw, root / 'missing', 'http://cpu-fixture.invalid', opener=rejected)
    assert failed['captured_task_responses'] == 0
    receipt = json.loads(next((root / 'missing/raw').glob('*.json')).read_text())
    assert receipt['HTTP_status'] == 404 and not receipt['complete_task_JSON']
    checks.append('missing_original_task_preserves_HTTP_body_without_fabricating_score')
    def stalled(request, timeout):
        time.sleep(1)
        raise AssertionError('Deadline failed')
    started = time.monotonic()
    bounded = collect(manifest_raw, root / 'bounded', 'http://cpu-fixture.invalid', seconds=.05, opener=stalled)
    assert time.monotonic() - started < .5 and bounded['captured_task_responses'] == 0
    assert bounded['errors'][0]['error_type'] == 'TimeoutError'
    checks.append('real_process_alarm_bounds_stalled_read_and_retains_partial_receipt')
    manifest_path = root / 'manifest.json'
    manifest_path.write_bytes(manifest_raw)
    report_path = root / 'summary.json'
    command = [sys.executable, str(HERE / 'evaluate_records.py'), 'aggregate', '--manifest', str(manifest_path),
               '--task-results', saved['immutable_index'], '--output', str(report_path)]
    result = subprocess.run(command, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    summary = json.loads(report_path.read_text())
    assert summary['accuracy'] == 1.0 and summary['known_objective_results'] == 1
    checks.append('actual_aggregation_CLI_reads_hash_bound_original_task_files')
    selection_path = root / 'selection.json'
    result = subprocess.run([sys.executable, str(HERE / 'evaluate_records.py'), 'select', '--summaries', str(report_path),
                             '--expected-steps', '5', '--output', str(selection_path)], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    selection = json.loads(selection_path.read_text())
    assert selection['selected_step'] == 5 and selection['selection_is_final']
    checks.append('actual_selection_CLI_uses_declared_schedule_and_counts')
    Path(record['path']).write_bytes(b'changed fixture')
    command[-1] = str(root / 'tampered.json')
    result = subprocess.run(command, capture_output=True, timeout=15)
    assert result.returncode != 0 and b'differs from its saved receipt' in result.stderr
    assert not (root / 'tampered.json').exists()
    checks.append('CLI_refuses_modified_original_response_before_writing_a_report')
proof = {'accepted': True, 'created': time.time(), 'checks': checks, 'count': len(checks),
         'actual_cli_integer_exits': [0, 0], 'intentional_corruption_refusal_checked': True,
         'actual_network_GPU_or_model_calls': 0, 'synthetic_fixture_only': True}
(HERE / 'COLLECTOR_CLI_CPU_ACCEPTED.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof, indent=2))
