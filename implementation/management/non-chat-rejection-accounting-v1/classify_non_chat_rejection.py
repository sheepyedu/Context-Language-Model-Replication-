"""Classify original non-chat payload schema errors without changing receipts."""

import hashlib
import json
from pathlib import Path


def classify_non_chat_rejection(root: Path, receipt: dict) -> dict | None:
    error_summary = receipt.get('error') or {}
    if not isinstance(error_summary, dict):
        return None
    if error_summary.get('type') != 'UpstreamHTTPError':
        return None
    if 'missing field `messages`' not in error_summary.get('message', ''):
        return None
    if any(receipt.get(field) is not None for field in
           ('input_tokens', 'output_tokens', 'total_tokens', 'model_reported', 'response_id')):
        return None
    scope = receipt.get('scope') or {}
    session_id = scope.get('session_id')
    call_id = receipt.get('call_id')
    if not isinstance(session_id, str) or not isinstance(call_id, str):
        return None
    attempt = root / 'runs/provider-attempts' / session_id / call_id
    if attempt.resolve().parent != (root / 'runs/provider-attempts' / session_id).resolve():
        return None
    if (root / 'runs/provider-attempts').resolve() not in attempt.resolve().parents:
        return None
    if str(attempt) != receipt.get('source_directory'):
        return None
    required = {name: attempt / name for name in ('INTENT.json', 'ERROR.json', 'TOKEN_RECEIPT.json')}
    if not all(path.is_file() for path in required.values()):
        return None
    if any(path.name not in required for path in attempt.iterdir()):
        return None
    raw = {name: path.read_bytes() for name, path in required.items()}
    intent = json.loads(raw['INTENT.json'])
    error = json.loads(raw['ERROR.json'])
    if json.loads(raw['TOKEN_RECEIPT.json']) != receipt:
        return None
    request = intent.get('request') or {}
    if not isinstance(request, dict):
        return None
    action = request.get('action')
    if not isinstance(action, str):
        return None
    field = {'search': 'query', 'fetch': 'url'}.get(action)
    if field is None or set(request) != {'action', field, 'model'}:
        return None
    if not isinstance(request[field], str) or not request[field]:
        return None
    if request['model'] != receipt.get('model_requested'):
        return None
    body = error.get('body') or {}
    if not isinstance(body, dict):
        return None
    body_error = body.get('error') or {}
    if not isinstance(body_error, dict):
        return None
    if (error.get('type') != 'UpstreamHTTPError'
            or body_error.get('code') != 'json_parse_error'
            or body_error.get('type') != 'invalid_request_error'
            or error.get('message') != error_summary.get('message')
            or body_error.get('message') != error.get('message')):
        return None
    if not isinstance(intent.get('created'), (int, float)):
        return None
    if not isinstance(error.get('finished'), (int, float)):
        return None
    if intent['created'] > error['finished']:
        return None
    return {
        'call_id': call_id,
        'classification': 'non-chat payload rejected by model-endpoint messages schema',
        'request_action': action,
        'usage': None,
        'caller_command_provenance': 'not asserted by this payload classification',
        'original_receipt_unchanged': True,
        'source_sha256': {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()},
    }
