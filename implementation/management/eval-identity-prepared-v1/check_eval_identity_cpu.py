"""Verify future experiment IDs with the original native evaluation producer."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import yaml

from prepare_eval_identity import prepare_config


ROOT = Path('/mount-data/CLM-Replication')
DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'management/evaluation-prepared-v1'))
from compile_manifest import compile_manifest
from question_metrics import require_distinct_task_namespace
from slime_bridge import rollout
from slime_bridge.config import resolve_polar_slime_config


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES', '') == ''
    base_path = ROOT / 'management/train-config-v7.yaml'
    original_bytes = base_path.read_bytes()
    original = yaml.safe_load(original_bytes)
    checks = []
    for invalid in [None, '', '   ', 1234]:
        try:
            prepare_config({**original, 'clm_run_id': invalid})
        except ValueError:
            pass
        else:
            raise AssertionError('An incomplete experiment identity was accepted')
    checks.append('empty_or_non_string_run_identity_rejected')
    config = prepare_config(original)
    assert config != original
    assert {key for key in original.keys() | config.keys()
            if original.get(key) != config.get(key)} == {'polar_task_id_template'}
    assert config['polar_task_template'] == original['polar_task_template']
    checks.append('only_task_ID_template_changes_with_all_task_and_evaluator_settings_preserved')

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        rows = [{'prompt': [{'role': 'user', 'content': f'Identity fixture question {i}'}],
                 'label': f'PRIVATE_FIXTURE_{i}', 'metadata': {'qid': qid}}
                for i, qid in enumerate([17, '17'])]
        data = directory / 'fixtures.jsonl'
        data.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        source = directory / 'source.yaml'
        source.write_text(yaml.safe_dump(original))
        output = directory / 'prepared.yaml'
        command = [sys.executable, str(DIRECTORY / 'prepare_eval_identity.py'),
                   '--config', str(source), '--output', str(output)]
        completed = subprocess.run(command, capture_output=True, timeout=15)
        assert completed.returncode == 0, completed.stderr.decode()
        assert yaml.safe_load(output.read_text()) == config
        refused = subprocess.run(command, capture_output=True, timeout=15)
        assert refused.returncode != 0
        assert source.read_text() == yaml.safe_dump(original)
        checks.append('actual_CLI_writes_new_configuration_and_preserves_existing_output_and_source')

        manifests = []
        run_ids = ['c1-seed1234', 'c1-seed1235', 'c1-seed1236',
                   'c2-seed1235', 'c2-seed1236']
        for run_id in run_ids:
            candidate = prepare_config({**original, 'clm_run_id': run_id})
            candidate['clm_phase'] = 'validation'
            path = directory / f'{run_id}.yaml'
            path.write_text(yaml.safe_dump(candidate))
            for step in [0, 5]:
                manifest = compile_manifest(path, data, 'fixture-dataset', step, 'validation')
                for previous in manifests:
                    require_distinct_task_namespace(manifest, previous)
                manifests.append(manifest)
            args = SimpleNamespace(**candidate, rollout_batch_size=1, n_samples_per_prompt=1)
            native = resolve_polar_slime_config(args)
            groups = [[SimpleNamespace(**row, index=i, group_index=i)]
                      for i, row in enumerate(rows)]
            captured = []

            async def submit(client, base_url, payload):
                captured.append(payload)
                return SimpleNamespace(results=[])

            with patch.object(rollout, '_submit_and_wait_for_task', side_effect=submit), \
                 patch.object(rollout, '_convert_task_result_to_samples', return_value=[]), \
                 patch.object(httpx.AsyncClient, 'post', side_effect=AssertionError('Actual HTTP forbidden')):
                asyncio.run(rollout._submit_eval_groups(
                    args=args, config=native, dataset_name='fixture-dataset',
                    rollout_id=5, sample_groups=groups))
            assert [payload['task_id'] for payload in captured] == [
                case['task_id'] for case in manifests[-1]['cases']]
            assert [payload['metadata']['qid'] for payload in captured] == [17, '17']
            assert all(payload['metadata']['condition'] == run_id for payload in captured)
            assert [payload['instruction'] for payload in captured] == [
                row['prompt'][0]['content'] for row in rows]
            assert all('PRIVATE_FIXTURE_' not in json.dumps(payload['agent'])
                       for payload in captured)
        checks.append('five_future_runs_and_two_steps_have_disjoint_original_native_task_IDs')
        checks.append('five_actual_native_eval_producer_calls_match_the_compiled_manifest')
        checks.append('typed_qids_and_condition_identity_preserved_with_policy_input_bytes_unchanged')
        checks.append('private_fixture_labels_remain_outside_policy_instruction_and_agent')

        actual_data = ROOT / 'data/validation-chat.jsonl'
        existing = json.loads((ROOT / 'management/evaluation-prepared-v1/'
                               'EXPECTED_C2_POLICY5_VALIDATION.json').read_text())
        full_manifests = [existing]
        for run_id in run_ids:
            path = directory / f'{run_id}.yaml'
            manifest = compile_manifest(path, actual_data, 'openresearcher', 5, 'validation')
            assert len(manifest['cases']) == 500
            assert [case['qid'] for case in manifest['cases']] == [
                case['qid'] for case in existing['cases']]
            for previous in full_manifests:
                require_distinct_task_namespace(manifest, previous)
            full_manifests.append(manifest)
        checks.append('all_2500_future_validation_IDs_disjoint_from_each_other_and_the_active_500_IDs')
    assert base_path.read_bytes() == original_bytes
    assert not getattr(sys.modules.get('torch'), 'cuda',
                       SimpleNamespace(is_initialized=lambda: False)).is_initialized()
    checks.append('active_configuration_unchanged_CUDA_uninitialized_and_actual_HTTP_zero')
    proof = {'accepted': True, 'created': time.time(), 'count': len(checks), 'checks': checks,
             'source_config_sha256': hashlib.sha256(original_bytes).hexdigest(),
             'future_run_ids': run_ids, 'actual_validation_questions_per_manifest': 500,
             'compiled_future_question_IDs': 2500, 'model_or_Judge_calls': 0,
             'native_submission_and_sample_conversion_are_CPU_stubs': True,
             'current_training_or_future_dispatch_modified': False}
    (DIRECTORY / 'CPU_ACCEPTED.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
