"""Snapshot immutable provider receipts by trajectory, role, task and policy."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

from clm_replication.accounting import aggregate
from classify_ancillary_receipt import classify_ancillary_receipt


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

summary = {
    'created': created,
    'all': aggregate(receipts),
    'by_role': {key: aggregate(value) for key, value in roles.items()},
    'by_request_kind': {key: aggregate(value) for key, value in request_kinds.items()},
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
fields = ['session_id', 'task_ids', 'policy_versions']
for role in ('rollout', 'judge'):
    fields.extend(role + '_' + name for name in (
        'calls', 'input_tokens', 'output_tokens', 'total_tokens', 'missing_total_calls'))
with (output / 'TRAJECTORIES.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for session_id, items in sorted(sessions.items()):
        row = {
            'session_id': session_id,
            'task_ids': json.dumps(sorted({item['scope'].get('task_id') or 'unprovided' for item in items})),
            'policy_versions': json.dumps(sorted({str(item['scope'].get('policy_version', 'unprovided')) for item in items})),
        }
        for role in ('rollout', 'judge'):
            subset = [item for item in items if item['role'] == role]
            count = aggregate(subset)
            row[role + '_calls'] = count['provider_attempts']
            for field in ('input_tokens', 'output_tokens', 'total_tokens'):
                # None is exported as an empty cell, preserving the known lower bound in SUMMARY.
                row[role + '_' + field] = count['complete_' + field]
            row[role + '_missing_total_calls'] = count['missing_total_tokens_calls']
        writer.writerow(row)
print(json.dumps({
    'output': str(output),
    'all': summary['all'],
    'by_role': summary['by_role'],
    'by_request_kind':summary['by_request_kind'],
    'trajectory_rows': len(sessions),
    'all_dimensions_reconcile': True,
    'reported_total_mismatches': len(reported_total_mismatches),
}, indent=2))
