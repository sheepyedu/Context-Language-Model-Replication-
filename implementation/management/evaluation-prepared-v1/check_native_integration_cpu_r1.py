"""Exercise the original eval producer and native result schema on CPU fixtures."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path('/mount-data/CLM-Replication')
sys.path.insert(0, str(ROOT / 'src/ProRL/src'))
import httpx
import yaml
from polar.rollout.models import TaskResult, TaskStatus
from slime_bridge import rollout
from slime_bridge.config import resolve_polar_slime_config
from compile_manifest import compile_manifest
from question_metrics import aggregate_questions

assert os.environ.get('CUDA_VISIBLE_DEVICES', '') == ''
checks = []
with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    rows = [{'prompt': [{'role': 'user', 'content': f'Generic fixture question {i}'}],
             'label': f'PRIVATE_FIXTURE_LABEL_{i}', 'metadata': {'qid': qid}}
            for i, qid in enumerate([17, '17'])]
    data = directory / 'questions.jsonl'
    data.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    config_path = ROOT / 'management/train-config-v7.yaml'
    expected = compile_manifest(config_path, data, 'fixture-dataset', 5, 'validation')
    config = yaml.safe_load(config_path.read_text())
    config['clm_phase'] = 'validation'
    args = SimpleNamespace(**config, rollout_batch_size=1, n_samples_per_prompt=1)
    native_config = resolve_polar_slime_config(args)
    groups = [[SimpleNamespace(**row, index=i, group_index=i)] for i, row in enumerate(rows)]
    captured = []
    original_results = {}
    async def submit(client, base_url, payload):
        captured.append(payload)
        position = payload['metadata']['group_id']
        qid = payload['metadata']['qid']
        success = position == 0
        task_id = payload['task_id']
        wire = {'task_id': task_id, 'status': 'completed', 'total_sessions': 1, 'completed_sessions': 1,
                'results': [{'session_id': f'fixture-session-{position}', 'task_id': task_id,
                             'status': 'COMPLETED', 'trajectory': {'status': 'completed', 'traces': [],
                             'metadata': {'task_metadata': {'qid': qid}, 'evaluation': {
                                 'outcome_reward': int(success), 'clm': {
                                     'success': success, 'judge': {'success': success}, 'flops': 100}}}}}]}
        original_results[task_id] = TaskStatus.model_validate(wire).model_dump(mode='json')
        return TaskResult.model_validate(wire)
    def samples(config, result, group, **kwargs):
        session = result.results[0]
        return [SimpleNamespace(reward={'score': session.trajectory.metadata['evaluation']['outcome_reward']},
                status='COMPLETED', metadata={'polar': {'session_id': session.session_id,
                'session_status': 'COMPLETED', 'placeholder': False}})]
    with patch.object(rollout, '_submit_and_wait_for_task', side_effect=submit), \
         patch.object(rollout, '_convert_task_result_to_samples', side_effect=samples), \
         patch.object(httpx.AsyncClient, 'post', side_effect=AssertionError('CPU fixture attempted actual HTTP')):
        asyncio.run(rollout._submit_eval_groups(args=args, config=native_config, dataset_name='fixture-dataset',
                                               rollout_id=5, sample_groups=groups))
    assert [payload['task_id'] for payload in captured] == [case['task_id'] for case in expected['cases']]
    checks.append('manifest_task_ids_equal_original_native_eval_producer')
    assert [payload['metadata']['qid'] for payload in captured] == [17, '17']
    checks.append('native_producer_preserves_typed_raw_question_ids')
    for payload in captured:
        assert payload['metadata']['purpose'] == 'validation'
        assert payload['metadata']['policy_version'] == 5
        assert 'PRIVATE_FIXTURE_LABEL' not in payload['instruction']
        assert 'PRIVATE_FIXTURE_LABEL' not in json.dumps(payload['agent'])
    checks.append('labels_are_not_in_native_policy_instruction_or_agent_configuration')
    report = aggregate_questions(expected, original_results)
    assert report['accuracy'] == .5 and report['known_objective_results'] == 2
    checks.append('native_TaskStatus_and_SessionResult_JSON_feed_question_accuracy')
    assert all(row['trajectory_trace_count'] == 0 for row in report['rows'])
    checks.append('fixture_zero_traces_are_explicit_without_claiming_model_execution')
assert not getattr(sys.modules.get('torch'), 'cuda', SimpleNamespace(is_initialized=lambda: False)).is_initialized()
checks.append('CUDA_not_initialized_and_actual_HTTP_guard_never_triggered')
proof = {'accepted': True, 'created': time.time(), 'checks': checks, 'count': len(checks),
         'synthetic_fixture_questions': 2, 'original_native_eval_producer_called': True,
         'HTTP_submit_and_trace_conversion_are_CPU_stubs': True,
         'actual_HTTP_GPU_or_model_calls': 0, 'actual_500_question_validation_pending': True}
Path(__file__).with_name('NATIVE_CPU_ACCEPTED.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof, indent=2))
