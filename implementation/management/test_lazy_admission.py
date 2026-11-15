"""Prove ordered-prefix equivalence and preserve the actual frozen payloads."""

from argparse import Namespace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from clm_replication.bridge import candidate_decision, should_admit_candidate


root = Path('/mount-data/CLM-Replication')
output = root / 'management/training-v6-recovery'
patterns = 0
for previously_started in (set(), set(range(9)), {12, 15}):
    for pattern in range(1 << 16):
        flags = [bool(pattern & (1 << position)) for position in range(16)]
        reference = [i for i, eligible in enumerate(flags) if eligible][:8]
        selected = []
        executed = []
        eligible_count = 0
        for position, eligible in enumerate(flags):
            if should_admit_candidate(eligible_count, 8, position in previously_started):
                executed.append(position)
                eligible_count += int(eligible)
                if eligible and len(selected) < 8:
                    selected.append(position)
        assert selected == reference
        assert previously_started.issubset(executed)
        assert executed == sorted(set(executed))
        patterns += 1

archives = [
    ('runs/c2-seed1234/audit/rollout-00000/candidate-000/group-000-result.json', False),
    ('management/group004-terminal-snapshot.json', True),
    ('management/group005-terminal-snapshot.json', True),
]
decisions = []
for name, expected in archives:
    data = json.loads((root / name).read_text())
    samples = []
    for session in data['results']:
        metadata = session['trajectory']['metadata']
        evaluation = metadata.get('evaluation') or {}
        # This tests the trajectory-level decision only. Token conversion and
        # the real closed-timeout training view have separate accepted audits.
        status = 'COMPLETED' if evaluation.get('clm') is not None else 'FAILED'
        samples.append(SimpleNamespace(status=SimpleNamespace(name=status), metadata={
            'polar': {'session_id': session['session_id'], 'trajectory_metadata': metadata},
        }))
    keep, reason, count = candidate_decision(Namespace(n_samples_per_prompt=32), {
        'samples': samples, 'rejection': None,
    })
    assert keep is expected
    decisions.append({'source': name, 'keep': keep, 'reason': reason, 'unique_outcomes': count})

from slime.utils.types import Sample
from slime_bridge import rollout
from slime_bridge.config import resolve_polar_slime_config

args = Namespace(**json.loads((output / 'validation-env/effective_args.json').read_text()))
assert args.clm_source_revision == 'training-v6'
assert args.rollout_batch_size == 8 and args.n_samples_per_prompt == 32
assert args.num_rollout == 70 and args.num_steps_per_rollout == 1
config = resolve_polar_slime_config(args)
directory = root / 'runs/c2-seed1234/audit/rollout-00000/candidate-000'
batch = json.loads((directory / 'BATCH.json').read_text())
payloads = 0
for intent in sorted(directory.glob('group-*-intent.json')):
    position = int(intent.name.split('-')[1])
    group = [Sample.from_dict(item) for item in batch['groups'][position]]
    payload = rollout._build_task_payload(args=args, config=config, group=group,
                                         rollout_id=0, task_position=position)
    payload['task_id'] = f'clm-{args.clm_run_id}-00000-000-{position:03d}'
    rollout._attach_scheduler_metadata(payload, group_id=group[0].group_index,
                                      policy_version=0, rollout_step=0)
    assert json.loads(intent.read_text()) == payload
    payloads += 1
proof = {
    'passed': True, 'exhaustive_selection_patterns': patterns,
    'candidate_positions_per_batch': 16, 'target_eligible_groups': 8,
    'same_first8_as_full16_filter': True, 'all_previous_intents_retained': True,
    'archived_metadata_decision_checks': decisions, 'existing_payloads_exact': payloads,
    'logical_trajectories_per_update': 256, 'target_updates': 70,
    'new_model_tool_or_judge_calls': 0,
    'source_sha256': hashlib.sha256((root / 'source-revisions/training-v6/clm_replication/bridge.py').read_bytes()).hexdigest(),
}
(output / 'LAZY_ADMISSION_CPU.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof, indent=2))
