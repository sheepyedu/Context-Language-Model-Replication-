"""Capture a read-only initial serving snapshot after native weight sync."""

import hashlib
import json
from pathlib import Path
import time
import urllib.request


def capture_initial_snapshot(args):
    run = Path(args.clm_audit_dir).parent
    owner = json.loads((run / 'CURRENT_OWNER.json').read_bytes())
    directory = run / f"attempt-{owner['attempt']:02d}" / 'initial-weight-snapshot'
    directory.mkdir(exist_ok=True)
    if (directory / 'ACCEPTED.json').exists():
        return
    with (directory / 'INTENT.json').open('x') as handle:
        json.dump({'created': time.time(), 'owner': owner,
                   'operation': 'native_read_only_snapshot_after_initial_weight_sync'}, handle)
    base = 'http://10.10.15.17:15000'
    with urllib.request.urlopen(base + '/model_info', timeout=15) as response:
        model = json.load(response)
    assert model['weight_version'] == '1'
    assert model['model_path'] == '/mount-data/Models/Qwen3.5-9B'
    payload = json.dumps({'action': 'snapshot'}).encode()
    (directory / 'request.json').write_bytes(payload)
    request = urllib.request.Request(base + '/weights_checker', data=payload,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        raw = response.read()
    (directory / 'response.raw').write_bytes(raw)
    assert json.loads(raw)['success'] is True
    proof = {'accepted': True, 'created': time.time(), 'owner': owner,
             'model': model, 'logical_policy': 0, 'new_generation_calls': 0,
             'response_sha256': hashlib.sha256(raw).hexdigest()}
    (directory / 'ACCEPTED.json').write_text(json.dumps(proof, indent=2))
