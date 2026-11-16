"""Anchor adjacent synchronized policy weights after the first real update."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path('/mount-data/CLM-Replication')
DIRECTORY = ROOT / 'management/update-chain-prepared-v1'
FIRST_HELPER_SHA = '75d9b08c1f2681060531ab220576b93393f2a2d365175378c6bedf87ba4121cb'
MODEL_URL = 'http://10.10.15.17:15000'


def load_first_helpers(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != FIRST_HELPER_SHA:
        raise ValueError('First-update helper differs from its accepted version')
    spec = importlib.util.spec_from_file_location('first_update_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_prefix(first, metrics, policy, model, identity, previous):
    if type(policy) is not int or policy < 1:
        raise ValueError('Policy must be a positive integer')
    if first.get('accepted') is not True or first.get('logical_policy') != 1:
        raise ValueError('The first real update has not been accepted')
    if first['owner_identity'] != identity:
        raise ValueError('The update chain crossed an owner identity boundary')
    if model.get('model_path') != '/mount-data/Models/Qwen3.5-9B':
        raise ValueError('Unexpected serving model')
    if model.get('weight_version') != str(policy + 1):
        raise ValueError('Serving version does not match the requested policy')
    first_step = first['native_metrics'][0]['train/step']
    if type(first_step) is not int or first_step < 0:
        raise ValueError('Initial native step must be a nonnegative integer')
    by_step = {}
    for row in metrics:
        step = row['train/step']
        if type(step) is not int:
            raise ValueError('Native step must be an integer')
        if step in by_step and by_step[step] != row:
            raise ValueError('Conflicting native metric records for the same step')
        by_step[step] = row
    expected = set(range(first_step, first_step + policy))
    if set(by_step) != expected:
        raise ValueError('Native optimizer-step coverage is incomplete or advanced')
    if policy == 1:
        if previous is not None:
            raise ValueError('The first anchor has no preceding chain anchor')
    elif (previous is None or previous.get('accepted') is not True
          or previous.get('logical_policy') != policy - 1
          or previous.get('raw_serving_version') != str(policy)
          or previous.get('owner_identity') != identity):
        raise ValueError('Adjacent accepted anchor is missing')
    return {'first_native_step': first_step, 'current_native_step': first_step + policy - 1,
            'unique_native_updates': len(by_step),
            'current_metric': by_step[first_step + policy - 1]}


def post_action(directory, name, action, opener=urllib.request.urlopen):
    if action not in {'snapshot', 'compare'}:
        raise ValueError('Only diagnostic snapshot and comparison are allowed')
    intent = directory / (name + '-INTENT.json')
    payload = json.dumps({'action': action}).encode()
    with intent.open('x') as handle:
        json.dump({'created': time.time(), 'action': action,
                   'request_sha256': hashlib.sha256(payload).hexdigest()}, handle, indent=2)
    (directory / (name + '-request.json')).write_bytes(payload)
    request = urllib.request.Request(MODEL_URL + '/weights_checker', data=payload,
                                    headers={'Content-Type': 'application/json'})
    started = time.time()
    status = None
    try:
        with opener(request, timeout=180) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as error:
        raw, status = error.read(), error.code
        (directory / (name + '-response.raw')).write_bytes(raw)
        (directory / (name + '-RECEIPT.json')).write_text(json.dumps({
            'started': started, 'finished': time.time(), 'HTTP_status': status,
            'response_sha256': hashlib.sha256(raw).hexdigest()}, indent=2))
        raise
    except Exception as error:
        partial = getattr(error, 'partial', b'')
        if isinstance(partial, bytes):
            (directory / (name + '-partial.raw')).write_bytes(partial)
        (directory / (name + '-RECEIPT.json')).write_text(json.dumps({
            'started': started, 'finished': time.time(), 'HTTP_status': status,
            'complete': False, 'outcome_unknown': True,
            'error_type': type(error).__name__}, indent=2))
        raise
    (directory / (name + '-response.raw')).write_bytes(raw)
    (directory / (name + '-RECEIPT.json')).write_text(json.dumps({
        'started': started, 'finished': time.time(), 'HTTP_status': status,
        'response_sha256': hashlib.sha256(raw).hexdigest()}, indent=2))
    if status != 200:
        raise ValueError('Native weight checker returned a non-200 response')
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=int, required=True)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--owner-start', type=int, required=True)
    args = parser.parse_args()
    first_path = ROOT / 'management/first-update-acceptance-v4/ACCEPTED.json'
    first = json.loads(first_path.read_text())
    helpers = load_first_helpers(ROOT / 'management/first_update_acceptance_v4.py')
    run = ROOT / 'runs/c2-seed1234'
    owner = json.loads((run / 'CURRENT_OWNER.json').read_text())
    assert owner['attempt'] == args.attempt
    launch = json.loads((Path(owner['directory']) / 'LAUNCH.json').read_text())
    identity = helpers.verify_owner_identity(owner, launch, args.owner_start)
    assert identity['uid'] == 0
    model = helpers.get_json(MODEL_URL + '/model_info')
    metrics = helpers.native_actor_metrics(Path(owner['directory']) / 'controller.log')
    previous_path = DIRECTORY / f'policy-{args.policy - 1:05d}/ACCEPTED.json'
    previous = json.loads(previous_path.read_text()) if args.policy > 1 else None
    native = validate_prefix(first, metrics, args.policy, model, identity, previous)
    if previous is not None:
        anchor = previous_path.parent / 'snapshot-response.raw'
        assert json.loads(anchor.read_text())['success'] is True
        assert hashlib.sha256(anchor.read_bytes()).hexdigest() == previous['snapshot_response_sha256']
    directory = DIRECTORY / f'policy-{args.policy:05d}'
    directory.mkdir(exist_ok=True)
    with (directory / 'INTENT.json').open('x') as handle:
        json.dump({'created': time.time(), 'logical_policy': args.policy,
                   'first_acceptance_sha256': hashlib.sha256(first_path.read_bytes()).hexdigest(),
                   'previous_acceptance_sha256': hashlib.sha256(previous_path.read_bytes()).hexdigest() if previous else None,
                   'owner_identity': identity, 'model_before': model,
                   'operation': 'native parameter comparison and CPU diagnostic snapshot only'}, handle, indent=2)
    (directory / 'native-metrics.json').write_text(json.dumps(metrics, indent=2))
    changes = first['changed_language_tensors']
    if args.policy > 1:
        comparison = post_action(directory, 'compare-previous', 'compare')
        changes = helpers.changed_language_tensors(comparison)
        assert helpers.get_json(MODEL_URL + '/model_info') == model
    snapshot = post_action(directory, 'snapshot', 'snapshot')
    assert snapshot.get('success') is True
    comparison = post_action(directory, 'compare-anchor', 'compare')
    assert comparison.get('success') is True
    assert helpers.get_json(MODEL_URL + '/model_info') == model
    assert json.loads((run / 'CURRENT_OWNER.json').read_text()) == owner
    assert helpers.verify_owner_identity(owner, launch, args.owner_start) == identity
    proof = {'accepted': True, 'created': time.time(), 'logical_policy': args.policy,
             'raw_serving_version': str(args.policy + 1), 'owner_identity': identity,
             'native_update': native, 'changed_language_tensors': changes,
             'snapshot_response_sha256': hashlib.sha256((directory / 'snapshot-response.raw').read_bytes()).hexdigest(),
             'native_metrics_sha256': hashlib.sha256((directory / 'native-metrics.json').read_bytes()).hexdigest(),
             'action_receipt_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in directory.glob('*-RECEIPT.json')},
             'new_generation_or_judge_calls': 0,
             'scope': 'native finite metrics, adjacent synchronized parameter changes and next CPU snapshot',
             'stability_complete': False,
             'mask_reward_accounting_and_checkpoint_acceptance_separate': True}
    (directory / 'ACCEPTED.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps({key: value for key, value in proof.items() if key != 'changed_language_tensors'}, indent=2))


if __name__ == '__main__':
    main()
