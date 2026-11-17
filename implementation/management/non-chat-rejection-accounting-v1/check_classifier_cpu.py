"""Exercise original-byte and schema boundaries with isolated CPU fixtures."""

import json
from pathlib import Path
import tempfile

from classify_non_chat_rejection import classify_non_chat_rejection


def fixture(root, action='search'):
    directory = root / 'runs/provider-attempts/session/call'
    directory.mkdir(parents=True)
    message = 'Invalid JSON data: missing field `messages`'
    receipt = {'call_id': 'call', 'scope': {'session_id': 'session'},
               'source_directory': str(directory), 'model_requested': 'test-model',
               'model_reported': None, 'response_id': None,
               'input_tokens': None, 'output_tokens': None, 'total_tokens': None,
               'error': {'type': 'UpstreamHTTPError', 'message': message}}
    request = {'action': action, 'model': 'test-model'}
    request['query' if action == 'search' else 'url'] = 'fixture'
    intent = {'created': 1, 'request': request}
    error = {'finished': 2, 'type': 'UpstreamHTTPError', 'message': message,
             'body': {'error': {'code': 'json_parse_error', 'type': 'invalid_request_error',
                                'message': message}}}
    for name, value in [('INTENT.json', intent), ('ERROR.json', error), ('TOKEN_RECEIPT.json', receipt)]:
        (directory / name).write_text(json.dumps(value))
    return directory, receipt, intent, error


checks = []
for action in ('search', 'fetch'):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        directory, receipt, intent, error = fixture(root, action)
        before = {path.name: path.read_bytes() for path in directory.iterdir()}
        result = classify_non_chat_rejection(root, receipt)
        assert result and result['usage'] is None and result['request_action'] == action
        assert before == {path.name: path.read_bytes() for path in directory.iterdir()}
        checks.append('valid_' + action)

for case in ('messages_present', 'wrong_action', 'wrong_model', 'wrong_error_code',
             'response_exists', 'raw_receipt_mismatch', 'known_usage', 'directory_escape',
             'error_before_request', 'missing_source', 'wrong_request_type', 'wrong_error_body_type'):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        directory, receipt, intent, error = fixture(root)
        if case == 'messages_present':
            intent['request']['messages'] = []
        elif case == 'wrong_action':
            intent['request']['action'] = 'generate'
        elif case == 'wrong_model':
            intent['request']['model'] = 'other'
        elif case == 'wrong_error_code':
            error['body']['error']['code'] = 'upstream_timeout'
        elif case == 'response_exists':
            (directory / 'RESPONSE.json').write_text('{}')
        elif case == 'raw_receipt_mismatch':
            receipt['scope']['qid'] = 123
        elif case == 'known_usage':
            receipt['total_tokens'] = 0
        elif case == 'directory_escape':
            receipt['scope']['session_id'] = '../../escape'
        elif case == 'error_before_request':
            error['finished'] = 0
        elif case == 'wrong_request_type':
            intent['request'] = ['unexpected']
        elif case == 'wrong_error_body_type':
            error['body']['error'] = ['unexpected']
        for name, value in [('INTENT.json', intent), ('ERROR.json', error)]:
            (directory / name).write_text(json.dumps(value))
        if case == 'missing_source':
            (directory / 'ERROR.json').unlink()
        assert classify_non_chat_rejection(root, receipt) is None, case
        checks.append(case)

print(json.dumps({'accepted': True, 'CPU_checks': len(checks), 'checks': checks,
                  'model_or_judge_calls': 0, 'original_receipts_modified': False}))
