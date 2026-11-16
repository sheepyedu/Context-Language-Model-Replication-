"""Compile expected task identities through the original native eval renderer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import yaml

ROOT = Path('/mount-data/CLM-Replication')
sys.path.insert(0, str(ROOT / 'src/ProRL/src'))
from polar.gateway.session import SessionStatus
from slime_bridge import rollout
from slime_bridge.config import resolve_polar_slime_config


def compile_manifest(config_path, data_path, dataset_name, checkpoint_step, purpose):
    config = yaml.safe_load(config_path.read_text())
    config['clm_phase'] = purpose
    args = SimpleNamespace(**config, rollout_batch_size=1, n_samples_per_prompt=1)
    native = resolve_polar_slime_config(args)
    data_raw = data_path.read_bytes()
    rows = [json.loads(line) for line in data_raw.decode().splitlines() if line.strip()]
    cases = []
    for position, row in enumerate(rows):
        sample = SimpleNamespace(prompt=row['prompt'], label=row['label'], metadata=row['metadata'],
                                 index=position, group_index=position)
        payload = rollout._build_task_payload(args=args, config=native, group=[sample],
                                              rollout_id=checkpoint_step, task_position=position)
        task_id = rollout._eval_task_id(payload['task_id'], dataset_name=dataset_name,
                                       rollout_id=checkpoint_step, position=position)
        cases.append({'qid': row['metadata']['qid'], 'task_id': task_id})
    return {'condition': config['clm_run_id'], 'purpose': purpose,
            'checkpoint_step': checkpoint_step, 'dataset_name': dataset_name,
            'dataset_path': str(data_path), 'dataset_sha256': hashlib.sha256(data_raw).hexdigest(),
            'config_path': str(config_path), 'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
            'native_renderer_source_sha256': hashlib.sha256(Path(rollout.__file__).read_bytes()).hexdigest(),
            'terminal_session_statuses': sorted(status.value for status in SessionStatus.terminal()),
            'sessions_per_question': 1, 'cases': cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--checkpoint-step', type=int, required=True)
    parser.add_argument('--purpose', choices=['validation', 'final_test'], required=True)
    parser.add_argument('--expected-count', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--avoid-manifests', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    if args.checkpoint_step < 0:
        raise ValueError('Checkpoint step must be nonnegative')
    manifest = compile_manifest(args.config, args.data, args.dataset_name, args.checkpoint_step, args.purpose)
    if len(manifest['cases']) != args.expected_count:
        raise ValueError('Dataset count differs from the original experiment plan')
    from question_metrics import aggregate_questions, require_distinct_task_namespace
    aggregate_questions(manifest, {})
    for path in args.avoid_manifests:
        require_distinct_task_namespace(manifest, json.loads(path.read_text()))
    with args.output.open('x') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'prepared_only': True, 'expected_questions': len(manifest['cases']),
                      'dataset_sha256': manifest['dataset_sha256'], 'model_calls': 0}, indent=2))


if __name__ == '__main__':
    main()
