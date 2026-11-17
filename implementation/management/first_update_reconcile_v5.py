"""Verify the original HTTP400 comparison from its native server log."""

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

from native_weight_evidence import changed_language_tensors


ROOT = Path('/mount-data/CLM-Replication')
FIRST_HELPER_SHA = '75d9b08c1f2681060531ab220576b93393f2a2d365175378c6bedf87ba4121cb'
MODEL_SOURCE_SHA = '60b00905037611c751b2c44f956c82b065a25001dc6a3b9b6f8bf37eb797c050'


def load_helpers(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != FIRST_HELPER_SHA:
        raise ValueError('Original native-metric and owner helpers changed')
    spec = importlib.util.spec_from_file_location('original_first_update', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_comparison(raw, intent):
    text = raw.decode(errors='replace')
    markers = re.findall(
        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) TP(\d+)\] '
        r'\[WeightChecker\] handle action=compare', text)
    expected_time = datetime.fromtimestamp(intent['created'], ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
    if len(markers) != 4 or {rank for _, rank in markers} != {'0', '1', '2', '3'}:
        raise ValueError('The original comparison must cover the four native TP ranks')
    if any(timestamp != expected_time for timestamp, _ in markers):
        raise ValueError('Native comparison timestamps differ from the original intent')
    if 'POST /weights_checker HTTP/1.1" 400 Bad Request' not in text:
        raise ValueError('Original native HTTP400 completion is missing')
    if 'handle action=snapshot' in text:
        raise ValueError('A diagnostic snapshot changed within the comparison')
    rows = changed_language_tensors({'success': False, 'message': text})
    if not any(row['name'] == 'lm_head.weight' for row in rows):
        raise ValueError('The source-identified language output head did not change')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--owner-start', type=int, required=True)
    args = parser.parse_args()
    directory = ROOT / 'management/first-update-reconciliation-v5'
    old = ROOT / 'management/first-update-acceptance-v4'
    intent_raw = (old / 'INTENT.json').read_bytes()
    intent = json.loads(intent_raw)
    manifest = json.loads((directory / 'ORIGINAL_COMPARE_SOURCE.json').read_text())
    raw = (directory / 'ORIGINAL_COMPARE_SERVER.log').read_bytes()
    assert hashlib.sha256(intent_raw).hexdigest() == manifest['original_intent_sha256']
    assert hashlib.sha256(raw).hexdigest() == manifest['raw_slice_sha256']
    assert manifest['observed_HTTP_status'] == 400 and manifest['client_raw_response_body_missing']
    with Path(manifest['source']).open('rb') as handle:
        handle.seek(manifest['byte_start'])
        assert handle.read(manifest['byte_end'] - manifest['byte_start']) == raw
        suffix = handle.read()
    assert b'[WeightChecker] handle action=' not in suffix
    model_source = ROOT / 'env/rl/lib/python3.11/site-packages/sglang/srt/models/qwen3_5.py'
    assert hashlib.sha256(model_source.read_bytes()).hexdigest() == MODEL_SOURCE_SHA
    assert 'head = self.lm_head.weight' in model_source.read_text()
    helpers = load_helpers(ROOT / 'management/first_update_acceptance_v4.py')
    run = ROOT / 'runs/c2-seed1234'
    owner = json.loads((run / 'CURRENT_OWNER.json').read_text())
    assert owner['attempt'] == args.attempt == intent['attempt']
    launch = json.loads((Path(owner['directory']) / 'LAUNCH.json').read_text())
    identity = helpers.verify_owner_identity(owner, launch, args.owner_start)
    assert identity['uid'] == 0
    model = helpers.get_json('http://10.10.15.17:15000/model_info')
    assert model == intent['model_before'] and model['weight_version'] == '2'
    assert model['architectures'] == ['Qwen3_5ForConditionalGeneration']
    metrics = helpers.native_actor_metrics(Path(owner['directory']) / 'controller.log')
    assert metrics and len({row['train/step'] for row in metrics}) == 1
    snapshot = Path(owner['directory']) / 'initial-weight-snapshot/response.raw'
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == intent['source_snapshot_sha256']
    assert json.loads(snapshot.read_text())['success'] is True
    changes = original_comparison(raw, intent)
    with (directory / 'INTENT.json').open('x') as handle:
        json.dump({'created': time.time(), 'operation': 'reconcile_original_server_log_only',
                   'original_intent_sha256': manifest['original_intent_sha256'],
                   'raw_log_sha256': manifest['raw_slice_sha256'], 'model_before': model}, handle, indent=2)
    assert helpers.get_json('http://10.10.15.17:15000/model_info') == model
    assert json.loads((run / 'CURRENT_OWNER.json').read_text()) == owner
    assert helpers.verify_owner_identity(owner, launch, args.owner_start) == identity
    proof = {'accepted': True, 'created': time.time(), 'attempt': args.attempt,
             'logical_policy': 1, 'raw_serving_version': '2', 'owner_identity': identity,
             'native_metrics': metrics, 'changed_language_tensors': changes,
             'evidence_type': 'original_native_server_comparison_exception_log',
             'original_compare_source': manifest, 'original_HTTP_response_body_missing': True,
             'original_v4_integer_exit': 1, 'new_comparison_or_generation_or_Judge_calls': 0,
             'language_role_source_sha256': MODEL_SOURCE_SHA,
             'scope': 'first native update with finite gradient/loss and actual synchronized language-weight changes',
             'checkpoint_restore_accepted': False, 'T06_complete': False}
    with (directory / 'ACCEPTED.json').open('x') as handle:
        json.dump(proof, handle, indent=2)
    print(json.dumps({key: value for key, value in proof.items()
                      if key not in {'native_metrics', 'changed_language_tensors'}}, indent=2))


if __name__ == '__main__':
    main()
