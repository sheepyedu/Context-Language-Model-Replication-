"""Verify the individually accepted non-generation HTTP rejection, without editing it."""
import hashlib
import json
from pathlib import Path


def classify_ancillary_receipt(root: Path, receipt: dict) -> dict | None:
    proof_path = root / 'management/request-routing-diagnostic/ACCEPTED.json'
    if not proof_path.exists():
        return None
    proof = json.loads(proof_path.read_bytes())
    if receipt['call_id'] != proof['call_id']:
        return None
    assert proof['accepted'] is True
    assert proof['classification'] == 'ancillary tool HTTP request rejected by model-endpoint JSON schema'
    sid = receipt['scope']['session_id']
    assert sid == proof['session_id']
    attempt = root / 'runs/provider-attempts' / sid / receipt['call_id']
    command_dir = root / 'runs/session-artifacts' / sid / 'logs/agent/commands/0077'
    sources = {name: path.read_bytes() for name, path in {
        'request_intent': attempt / 'INTENT.json',
        'error': attempt / 'ERROR.json',
        'token_receipt': attempt / 'TOKEN_RECEIPT.json',
        'command_intent': command_dir / 'INTENT.json',
        'command_result': command_dir / 'RESULT.json',
    }.items()}
    intent = json.loads(sources['request_intent'])
    error = json.loads(sources['error'])
    command = json.loads(sources['command_intent'])
    result = json.loads(sources['command_result'])
    assert json.loads(sources['token_receipt']) == receipt
    request = intent['request']
    assert set(request) == {'action', 'query', 'model'}
    assert request['action'] == 'search' and isinstance(request['query'], str)
    assert error['type'] == 'UpstreamHTTPError'
    assert error['body']['error']['code'] == 'json_parse_error'
    assert 'missing field `messages`' in error['message']
    assert not (attempt / 'RESPONSE.json').exists()
    assert all(receipt[field] is None for field in ('input_tokens', 'output_tokens', 'total_tokens'))
    assert ':19171/v1' in command['command'] and "base + '/clm/web'" in command['command']
    assert request['query'] in command['command'] and sid in command['command']
    assert command['started'] <= intent['created'] <= error['finished'] <= result['finished']
    assert result['return_code'] == 1 and result['timed_out'] is False
    return {
        'call_id': receipt['call_id'], 'classification': proof['classification'],
        'usage': None, 'original_receipt_unchanged': True,
        'source_sha256': {name: hashlib.sha256(value).hexdigest() for name, value in sources.items()},
        'accepted_diagnostic_sha256': hashlib.sha256(proof_path.read_bytes()).hexdigest(),
    }
