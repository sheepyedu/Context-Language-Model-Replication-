"""Snapshot immutable provider receipts by trajectory, role, task and policy."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

from clm_replication.accounting import aggregate
from classify_ancillary_receipt import classify_ancillary_receipt
from classify_non_chat_rejection import classify_non_chat_rejection


root = Path('/mount-data/CLM-Replication')
ledger = root / 'runs/token-ledger'
created = time.time()
receipts = []
hashes = {}
for path in sorted(ledger.glob('*.json')):
    content = path.read_bytes()
    receipt = json.loads(content)
    assert path.stem == receipt['call_id']
    hashes[path.name] = hashlib.sha256(content).hexdigest()
    receipts.append(receipt)
assert len({item['call_id'] for item in receipts}) == len(receipts)

request_kinds = defaultdict(list)
ancillary_dispositions=[]
for receipt in receipts:
    disposition=classify_ancillary_receipt(root,receipt)
    kind='provider_model_request'
    if disposition is not None:
        kind='accepted_ancillary_http_schema_rejection'
        ancillary_dispositions.append(disposition)
    else:
        disposition = classify_non_chat_rejection(root, receipt)
        if disposition is not None:
            kind = 'non_chat_payload_schema_rejection'
            ancillary_dispositions.append(disposition)
    request_kinds[kind].append(receipt)

sessions = defaultdict(list)
roles = defaultdict(list)
tasks = defaultdict(list)
policies = defaultdict(list)
reported_total_mismatches = []
for item in receipts:
    scope = item['scope']
    sessions[scope.get('session_id') or 'engineering_or_unattributed'].append(item)
    roles[item['role']].append(item)
    tasks[scope.get('task_id') or 'engineering_or_unattributed'].append(item)
    policies[str(scope.get('policy_version', 'unprovided'))].append(item)
    reported = item['reported_total_tokens']
    if reported is not None and item['total_tokens'] is not None and reported != item['total_tokens']:
        reported_total_mismatches.append(item['call_id'])

known_no_model = {}
zero_proof_path = root / 'management/queue-timeout-recovery-v1/PRE_DISPATCH_TIMEOUT_ROOT_ACCEPTED.json'
if zero_proof_path.exists():
    zero_proof = json.loads(zero_proof_path.read_bytes())
    assert zero_proof['accepted'] and zero_proof['model_intents'] == 0
    source = root / 'management/queue-timeout-recovery-v1/BEFORE_DRAIN_TASK_SESSIONS.json'
    rows = json.loads(source.read_bytes())['tasks']
    task_for_session = {item['session_id']: task['task']['task_id']
                        for task in rows for item in task['sessions'] if item['status'] == 'TIMEOUT'}
    intents = {}
    for path in (root / 'runs/c2-seed1234/audit').glob('rollout-*/candidate-*/group-*-intent.json'):
        request = json.loads(path.read_bytes())
        intents[request['task_id']] = request
    for sid in zero_proof['sessions']:
        assert not sessions[sid], 'A terminal known-unsent session acquired model receipts'
        assert not list((root / 'runs/provider-attempts' / sid).glob('*/INTENT.json'))
        task_id = task_for_session[sid]
        request = intents[task_id]
        policy = str(request['metadata']['policy_version'])
        tasks[task_id]  # Include the native task even when it used no provider calls.
        policies[policy]
        known_no_model[sid] = {'task_id': task_id, 'policy_version': policy, 'native_status': 'TIMEOUT',
                               'evidence': str(zero_proof_path), 'task_outcome_reward': None}

summary = {
    'created': created,
    'all': aggregate(receipts),
    'by_role': {key: aggregate(value) for key, value in roles.items()},
    'by_request_kind': {key: aggregate(value) for key, value in request_kinds.items()},
    'known_no_model_trajectories':known_no_model,
    'ancillary_dispositions':ancillary_dispositions,
    'by_session': {key: aggregate(value) for key, value in sessions.items()},
    'by_task': {key: aggregate(value) for key, value in tasks.items()},
    'by_policy': {key: aggregate(value) for key, value in policies.items()},
    'reported_total_mismatch_call_ids': reported_total_mismatches,
    'receipt_sha256': hashes,
    'coverage': 'Closed canonical receipts only. Calls still in flight are excluded until closure.',
    'counting_rule': 'Input includes cache, output includes reasoning. Reused training-token occurrences are separate.',
}
for dimension in ('by_role', 'by_session', 'by_task', 'by_policy', 'by_request_kind'):
    for field in ('provider_attempts', 'known_input_tokens', 'known_output_tokens', 'known_total_tokens'):
        assert sum(item[field] for item in summary[dimension].values()) == summary['all'][field]

output = root / 'runs/token-reports' / str(time.time_ns())
output.mkdir(parents=True)
(output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2))
fields = ['session_id', 'task_ids', 'policy_versions', 'related_source_sessions',
          'record_kind', 'native_status', 'known_no_model_evidence']
role_names = ['all'] + sorted(roles)
metrics = ('provider_attempts', 'known_input_tokens', 'complete_input_tokens',
           'known_output_tokens', 'complete_output_tokens', 'known_total_tokens',
           'complete_total_tokens', 'missing_total_tokens_calls')
for role in role_names:
    fields.extend(role + '_' + name for name in metrics)
csv_totals = defaultdict(int)
with (output / 'TRAJECTORIES.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for session_id, items in sorted(sessions.items()):
        row = {
            'session_id': session_id,
            'task_ids': json.dumps(sorted({item['scope'].get('task_id') or 'unprovided' for item in items}) if items else [known_no_model[session_id]['task_id']]),
            'policy_versions': json.dumps(sorted({str(item['scope'].get('policy_version', 'unprovided')) for item in items}) if items else [known_no_model[session_id]['policy_version']]),
            'related_source_sessions': json.dumps(sorted({item['scope']['source_session'] for item in items if item['scope'].get('source_session')})),
            'record_kind':'terminal_without_model_request' if session_id in known_no_model else 'provider_receipts',
            'native_status':known_no_model.get(session_id,{}).get('native_status'),
            'known_no_model_evidence':known_no_model.get(session_id,{}).get('evidence'),
        }
        for role in role_names:
            subset = items if role == 'all' else [item for item in items if item['role'] == role]
            count = aggregate(subset)
            for field in metrics:
                row[role + '_' + field] = count[field]
                if field in ('provider_attempts', 'known_input_tokens', 'known_output_tokens', 'known_total_tokens'):
                    csv_totals[role + '_' + field] += count[field]
        writer.writerow(row)
for role in role_names:
    expected = summary['all'] if role == 'all' else summary['by_role'][role]
    for field in ('provider_attempts', 'known_input_tokens', 'known_output_tokens', 'known_total_tokens'):
        assert csv_totals[role + '_' + field] == expected[field]
summary['csv_all_roles_reconcile'] = True
summary['csv_roles'] = role_names
(output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({
    'output': str(output),
    'all': summary['all'],
    'by_role': summary['by_role'],
    'by_request_kind':summary['by_request_kind'],
    'trajectory_rows': len(sessions),
    'known_no_model_trajectory_rows':len(known_no_model),
    'all_dimensions_reconcile': True,
    'csv_all_roles_reconcile':True,
    'csv_roles':role_names,
    'reported_total_mismatches': len(reported_total_mismatches),
}, indent=2))
