"""Validate the public CLM archive without importing training dependencies."""

import ast
import csv
import hashlib
import json
from pathlib import Path
import re


def validate(directory):
    directory = Path(directory).resolve()
    checks = []
    manifest = json.loads((directory / 'MANIFEST.json').read_text())
    actual_files = {str(path.relative_to(directory)) for path in directory.rglob('*')
                    if path.is_file()
                    and str(path.relative_to(directory)) not in {'MANIFEST.json', 'VALIDATION.json'}}
    assert actual_files == set(manifest['files'])
    checks.append('complete public file coverage including nested manifests')
    for relative, expected in manifest['files'].items():
        path = (directory / relative).resolve()
        assert path.is_relative_to(directory), relative
        raw = path.read_bytes()
        assert len(raw) == expected['bytes'], relative
        assert hashlib.sha256(raw).hexdigest() == expected['sha256'], relative
    checks.append('published file sizes and SHA256')
    python_files = list(directory.rglob('*.py'))
    for path in python_files:
        ast.parse(path.read_text(), filename=str(path))
    checks.append('all archived Python syntax')
    for path in directory.rglob('*.json'):
        json.loads(path.read_text())
    checks.append('all archived JSON structures')
    for path in directory.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if target.startswith(('https://', 'http://', '#')):
                continue
            assert (path.parent / target.split('#')[0]).exists(), (path.name, target)
    checks.append('archive documentation relative links')
    state = json.loads((directory / 'ARCHIVE_STATUS.json').read_text())
    assert state['formal_updates_accepted'] == 1
    assert state['formal_updates_planned'] == 70
    assert state['scientific_complete'] is False
    assert state['second_update_committed'] is False
    assert state['full_training_checkpoint_saved'] is False
    checks.append('partial scientific completion disclosed')
    stop = json.loads((directory / 'evidence/STOP_RESULT.json').read_text())
    assert not stop['remaining_matching_live_pids']
    assert stop['actual_owner_exit']['exit_code'] == -15
    assert stop['formal_updates'] == 1
    assert stop['training_complete'] is False
    gpu_lines = stop['gpus_after'].splitlines()
    assert len(gpu_lines) == 4 and all('0 MiB' in line for line in gpu_lines)
    checks.append('actual owner exit and four-card release')
    tokens = json.loads((directory / 'evidence/TOKEN_ACCOUNTING.json').read_text())
    total = tokens['all']
    assert total['known_input_tokens'] + total['known_output_tokens'] == total['known_total_tokens']
    assert sum(v['known_total_tokens'] for v in tokens['by_role'].values()) == total['known_total_tokens']
    assert total['known_total_tokens'] == 288093184
    assert tokens['by_role']['judge']['provider_attempts'] == 568
    assert tokens['by_role']['judge']['known_total_tokens'] == 623240
    assert total['missing_total_tokens_calls'] == 14 and total['money'] is None
    checks.append('known token and role totals with original nulls')
    with (directory / 'evidence/TOKEN_TRAJECTORIES_PUBLIC.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 771
    assert sum(int(row['all_known_total_tokens']) for row in rows) == total['known_total_tokens']
    checks.append('771 public accounting rows reconcile')
    uuid_pattern = re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')
    assert not uuid_pattern.search(json.dumps(tokens) + json.dumps(rows))
    assert tokens['known_no_model_trajectory_count'] == 91
    checks.append('public accounting identifiers anonymized and no-model count explicit')
    first = json.loads((directory / 'evidence/FIRST_UPDATE_DATA_ACCEPTED.json').read_text())
    assert first['accepted'] and first['logical_trajectories'] == 256
    assert first['counts']['samples'] == 11095
    assert len(first['groups']) == 8 and all(g['trajectories'] == 32 for g in first['groups'])
    assert sum(g['successes'] for g in first['groups']) == 43
    assert first['counts']['nonzero_efficiency_role_tokens'] == 10697
    checks.append('native first-update trajectory and efficiency counts')
    package = directory / 'framework/source-revisions/training-v10/clm_replication'
    for name in ['agent_runner.py', 'bridge.py', 'rewards.py', 'advantages.py',
                 'evaluator.py', 'judge.py', 'accounting.py', 'weight_provenance.py']:
        assert (package / name).is_file(), name
    checks.append('actual consumer source entrypoints present')
    launcher = (directory / 'framework/management/train-v14.sh').read_text()
    for argument in ['--num-rollout 70', '--rollout-batch-size 8', '--n-samples-per-prompt 32',
                     '--global-batch-size 256', '--tensor-model-parallel-size 4', '--bf16']:
        assert argument in launcher, argument
    checks.append('actual launcher scientific configuration')
    assert state['automatic_supervision_stopped'] and state['goal_cleared_by_user']
    checks.append('archive control state')
    return {'accepted': True, 'checks': checks, 'check_count': len(checks),
            'manifest_files': len(manifest['files']), 'python_files': len(python_files),
            'new_model_tool_judge_calls': 0, 'scope': 'public artifact consistency and syntax only'}


if __name__ == '__main__':
    print(json.dumps(validate(Path(__file__).parent), ensure_ascii=False, indent=2))
