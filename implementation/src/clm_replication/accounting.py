"""One immutable token receipt per real provider attempt, including judges."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import uuid


ROOT = Path('/mount-data/CLM-Replication')


def token_counts(response):
    response = response or {}
    usage = response.get('usage') or {}
    choices = response.get('choices') or [{}]
    choice = choices[0]
    prompt_ids = choice.get('input_token_ids', response.get('prompt_token_ids'))
    output_ids = choice.get('token_ids')
    reported_input = usage.get('prompt_tokens', usage.get('input_tokens'))
    reported_output = usage.get('completion_tokens', usage.get('output_tokens'))
    input_tokens = reported_input if reported_input is not None else (len(prompt_ids) if isinstance(prompt_ids, list) else None)
    output_tokens = reported_output if reported_output is not None else (len(output_ids) if isinstance(output_ids, list) else None)
    for value in (input_tokens, output_tokens):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError('Invalid token counter')
    input_details = usage.get('prompt_tokens_details') or usage.get('input_tokens_details') or {}
    output_details = usage.get('completion_tokens_details') or usage.get('output_tokens_details') or {}
    return {
        'input_tokens': input_tokens, 'output_tokens': output_tokens,
        'total_tokens': input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None,
        'reported_total_tokens': usage.get('total_tokens'),
        'cached_input_tokens_subset': input_details.get('cached_tokens'),
        'reasoning_output_tokens_subset': output_details.get('reasoning_tokens'),
        'input_count_source': 'usage' if reported_input is not None else ('token_ids' if input_tokens is not None else None),
        'output_count_source': 'usage' if reported_output is not None else ('token_ids' if output_tokens is not None else None),
    }


def record_tokens(directory, *, call_id, role, model_requested, response=None, scope=None, error=None):
    directory = Path(directory)
    receipt = {
        'call_id': call_id, 'role': role, 'model_requested': model_requested,
        'model_reported': (response or {}).get('model'), 'response_id': (response or {}).get('id'),
        'scope': scope or {}, 'source_directory': str(directory), 'error': error,
        **token_counts(response), 'money': None,
    }
    ledger = ROOT / 'runs/token-ledger'
    ledger.mkdir(parents=True, exist_ok=True)
    target = ledger / f'{call_id}.json'
    temporary = ledger / f'.{call_id}-{uuid.uuid4().hex}.tmp'
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    try:
        os.link(temporary, target)
    except FileExistsError:
        if json.loads(target.read_text()) != receipt:
            raise ValueError('Conflicting receipts for a provider attempt')
    finally:
        temporary.unlink()
    (directory / 'TOKEN_RECEIPT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def aggregate(receipts):
    result = {'provider_attempts': len(receipts)}
    for field in ('input_tokens', 'output_tokens', 'total_tokens'):
        values = [item[field] for item in receipts]
        result['known_' + field] = sum(value for value in values if value is not None)
        result['complete_' + field] = sum(values) if all(value is not None for value in values) else None
        result['missing_' + field + '_calls'] = sum(value is None for value in values)
    for field in ('cached_input_tokens_subset', 'reasoning_output_tokens_subset'):
        result['known_' + field] = sum(item[field] for item in receipts if item[field] is not None)
        result['missing_' + field + '_calls'] = sum(item[field] is None for item in receipts)
    result['money'] = None
    return result


def summarize(root):
    receipts = [json.loads(path.read_text()) for path in (root / 'runs/token-ledger').glob('*.json')]
    if len({item['call_id'] for item in receipts}) != len(receipts):
        raise ValueError('Duplicate canonical accounting identity')
    by_role = defaultdict(list)
    by_task = defaultdict(list)
    for receipt in receipts:
        by_role[receipt['role']].append(receipt)
        by_task[receipt['scope'].get('task_id', 'engineering_or_unattributed')].append(receipt)
    result = {'all': aggregate(receipts),
              'by_role': {key: aggregate(value) for key, value in by_role.items()},
              'by_task': {key: aggregate(value) for key, value in by_task.items()},
              'counting_rule': 'Input includes its cached subset. Output includes its reasoning subset. Subsets are not added again.',
              'coverage': 'All canonical attempt receipts, including discarded groups, failures, evaluation and engineering.'}
    output = root / 'runs/TOKEN_ACCOUNTING.json'
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(output)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))
