"""Check version adjacency and durable one-shot diagnostics without networking."""
import ast
from copy import deepcopy
from contextlib import redirect_stdout
import http.client
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import urllib.error
from unittest.mock import patch

import accept_update_chain as module

identity = {'pid': 123, 'uid': 0, 'start_ticks': 456, 'raw_sha256': 'fixture'}
row0 = {'train/step': 0, 'train/grad_norm': 1.0, 'train/loss': .1}
row1 = {'train/step': 1, 'train/grad_norm': .9, 'train/loss': .09}
first = {'accepted': True, 'logical_policy': 1, 'owner_identity': identity,
         'native_metrics': [row0]}
model = {'weight_version': '2', 'model_path': '/mount-data/Models/Qwen3.5-9B'}
previous = {'accepted': True, 'logical_policy': 1, 'raw_serving_version': '2',
            'owner_identity': identity}
checks = []
original_source = Path(__file__).resolve().parents[1] / 'update-chain-prepared-v1/accept_update_chain.py'
def prefix_ast(path):
    return ast.dump(next(node for node in ast.parse(path.read_text()).body
                         if isinstance(node, ast.FunctionDef) and node.name == 'validate_prefix'))
assert prefix_ast(original_source) == prefix_ast(Path(module.__file__))
checks.append('original_version_owner_and_native_step_prefix_gate_AST_unchanged')
assert module.validate_prefix(first, [row0], 1, model, identity, None)['unique_native_updates'] == 1
checks.append('first_anchor_requires_accepted_first_real_update')
model2 = {**model, 'weight_version': '3'}
assert module.validate_prefix(first, [row0, row1], 2, model2, identity, previous)['current_native_step'] == 1
checks.append('adjacent_second_update_accepted')
assert module.validate_prefix(first, [row0, row0, row1], 2, model2, identity, previous)['unique_native_updates'] == 2
checks.append('duplicate_identical_metric_rows_count_once')
first_one = deepcopy(first)
first_one['native_metrics'] = [{**row0, 'train/step': 1}]
assert module.validate_prefix(first_one, [row1], 1, model, identity, None)['first_native_step'] == 1
checks.append('native_first_step_number_is_preserved')

bad = [
    ({'accepted': False, **{k: v for k, v in first.items() if k != 'accepted'}}, [row0], 1, model, identity, None),
    (first, [row0], 0, model, identity, None),
    (first, [row0], True, model, identity, None),
    (first, [row0], 1, {**model, 'weight_version': '1'}, identity, None),
    (first, [row0], 1, {**model, 'model_path': '/different'}, identity, None),
    (first, [row0], 1, model, {**identity, 'start_ticks': 457}, None),
    (first, [row0], 2, model2, identity, previous),
    (first, [row0, {**row1, 'train/step': 2}], 2, model2, identity, previous),
    (first, [row0, row1], 2, model2, identity, None),
    (first, [row0, row1], 2, model2, identity, {**previous, 'raw_serving_version': '1'}),
    (first, [row0, row1], 2, model2, identity, {**previous, 'owner_identity': {}}),
    (first, [row0, {**row0, 'train/loss': .2}], 1, model, identity, None),
    (first, [{**row0, 'train/step': .5}], 1, model, identity, None),
]
for index, args in enumerate(bad):
    try:
        module.validate_prefix(*args)
    except ValueError:
        pass
    else:
        raise AssertionError(index)
checks.append('thirteen_invalid_acceptance_or_version_prefixes_rejected')


class Response:
    status = 200
    def __init__(self, body=b'{"success":true}'):
        self.body = body
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self):
        return self.body


with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    calls = []
    def opener(request, timeout):
        assert timeout == 180
        calls.append(json.loads(request.data)['action'])
        return Response()
    assert module.post_action(directory, 'snapshot', 'snapshot', opener)['success']
    assert calls == ['snapshot']
    assert (directory / 'snapshot-response.raw').read_bytes() == b'{"success":true}'
    checks.append('complete_native_response_and_intent_saved')
    try:
        module.post_action(directory, 'snapshot', 'snapshot', opener)
    except FileExistsError:
        pass
    else:
        raise AssertionError('Same action was replayed')
    assert calls == ['snapshot']
    checks.append('existing_intent_prevents_repeated_action')
    def incomplete(request, timeout):
        raise http.client.IncompleteRead(b'{"partial":', 12)
    try:
        module.post_action(directory, 'incomplete', 'compare', incomplete)
    except http.client.IncompleteRead:
        pass
    else:
        raise AssertionError('Partial native response accepted')
    assert (directory / 'incomplete-partial.raw').read_bytes() == b'{"partial":'
    assert json.loads((directory / 'incomplete-RECEIPT.json').read_text())['outcome_unknown']
    checks.append('transport_failure_retains_partial_and_unknown_without_acceptance')
    def rejected(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 500, 'fixture', {}, io.BytesIO(b'failed'))
    try:
        module.post_action(directory, 'rejected', 'compare', rejected)
    except urllib.error.HTTPError:
        pass
    else:
        raise AssertionError('HTTP failure accepted')
    assert (directory / 'rejected-response.raw').read_bytes() == b'failed'
    checks.append('HTTP_failure_body_and_integer_status_preserved')
    invalid_body = b'{"success":false,"message":"unrelated native error"}'
    def invalid_400(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 400, 'fixture', {}, io.BytesIO(invalid_body))
    try:
        module.post_action(directory, 'invalid400', 'compare', invalid_400)
    except ValueError:
        pass
    else:
        raise AssertionError('Unrelated HTTP400 accepted as a parameter change')
    assert (directory / 'invalid400-response.raw').read_bytes() == invalid_body
    assert json.loads((directory / 'invalid400-RECEIPT.json').read_text())['HTTP_status'] == 400
    checks.append('unrelated_HTTP400_body_retained_and_rejected')
    class BrokenBody(io.BytesIO):
        def read(self, *args):
            raise http.client.IncompleteRead(b'{"partial400":', 10)
    def partial_400(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 400, 'fixture', {}, BrokenBody())
    try:
        module.post_action(directory, 'partial400', 'compare', partial_400)
    except http.client.IncompleteRead:
        pass
    else:
        raise AssertionError('Incomplete HTTP400 body accepted')
    assert (directory / 'partial400-partial.raw').read_bytes() == b'{"partial400":'
    assert json.loads((directory / 'partial400-RECEIPT.json').read_text())['outcome_unknown']
    checks.append('partial_HTTP400_preserves_original_bytes_and_unknown_receipt')
    try:
        module.post_action(directory, 'forbidden', 'reset_tensors', opener)
    except ValueError:
        pass
    else:
        raise AssertionError('Parameter mutation operation was accepted')
    assert not (directory / 'forbidden-INTENT.json').exists()
    checks.append('parameter_mutation_action_refused_before_intent')
    helpers = module.load_first_helpers(Path(__file__).resolve().parents[1] / 'first_update_acceptance_v4.py')
    metrics_path = directory / 'native-controller-fixture.log'
    metrics_path.write_text("step 0: {'train/step': 0, 'train/grad_norm': 1.0, 'train/loss': .1}\n")
    assert helpers.native_actor_metrics(metrics_path) == [row0]
    checks.append('reused_pinned_native_finite_metric_parser')
    invalid_rows = [
        "{'train/step': 0, 'train/grad_norm': 0.0, 'train/loss': .1}",
        "{'train/step': 0, 'train/grad_norm': -1.0, 'train/loss': .1}",
        "{'train/step': 0, 'train/grad_norm': 1e309, 'train/loss': .1}",
        "{'train/step': 0, 'train/grad_norm': 1.0, 'train/loss': 1e309}",
        "{'train/step': 0, 'train/grad_norm': 1.0}",
    ]
    for row in invalid_rows:
        metrics_path.write_text('step 0: ' + row + '\n')
        try:
            helpers.native_actor_metrics(metrics_path)
        except ValueError:
            pass
        else:
            raise AssertionError(row)
    checks.append('reused_parser_rejects_zero_negative_nonfinite_and_missing_loss')
    root = directory / 'main-fixture'
    chain = root / 'management/update-chain-prepared-v2'
    chain.mkdir(parents=True)
    with patch.object(module, 'ROOT', root), patch.object(module, 'DIRECTORY', chain), \
         patch('sys.argv', ['helper', '--policy', '1', '--attempt', '13', '--owner-start', '456']), \
         patch.object(module, 'load_first_helpers') as loader:
        try:
            module.main()
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('Missing first acceptance was admitted')
        loader.assert_not_called()
    assert not list(chain.glob('policy-*/INTENT.json'))
    checks.append('main_refuses_missing_first_acceptance_before_HTTP_or_intent')
    first_directory = root / 'management/first-update-reconciliation-v5'
    first_directory.mkdir()
    changed = [{'name': 'language_model.fixture.weight', 'max_abs_err': .1, 'mean_abs_err': .01}]
    (first_directory / 'ACCEPTED.json').write_text(json.dumps({**first, 'changed_language_tensors': changed}))
    attempt = root / 'runs/c2-seed1234/attempt-13'
    attempt.mkdir(parents=True)
    owner = {'pid': 123, 'attempt': 13, 'argv': ['python', 'owner.py'], 'directory': str(attempt)}
    (attempt / 'LAUNCH.json').write_text(json.dumps(owner))
    (attempt.parent / 'CURRENT_OWNER.json').write_text(json.dumps(owner))
    original_post = module.post_action
    def fake_post(directory, name, action):
        response = {'success': True}
        if name == 'compare-previous':
            response = {'success': False, 'message': 'check tensor equality failed:\nname=language_model.fixture.weight max_abs_err=0.1 mean_abs_err=0.01'}
        def native_response(request, timeout):
            raw = json.dumps(response).encode()
            if name == 'compare-previous':
                raise urllib.error.HTTPError(request.full_url, 400, 'native inequality', {}, io.BytesIO(raw))
            return Response(raw)
        return original_post(directory, name, action, native_response)
    for policy, model_info, native_metrics in [(1, model, [row0]), (2, model2, [row0, row1])]:
        fake_helpers = SimpleNamespace(
            verify_owner_identity=lambda *args: identity,
            get_json=lambda url: model_info,
            native_actor_metrics=lambda path: native_metrics,
            changed_language_tensors=helpers.changed_language_tensors)
        with patch.object(module, 'ROOT', root), patch.object(module, 'DIRECTORY', chain), \
             patch('sys.argv', ['helper', '--policy', str(policy), '--attempt', '13', '--owner-start', '456']), \
             patch.object(module, 'load_first_helpers', return_value=fake_helpers), \
             patch.object(module, 'post_action', side_effect=fake_post), redirect_stdout(io.StringIO()):
            module.main()
        accepted = json.loads((chain / f'policy-{policy:05d}/ACCEPTED.json').read_text())
        assert accepted['logical_policy'] == policy and accepted['accepted']
        assert accepted['stability_complete'] is False
        assert accepted['changed_language_tensors'] == changed
        assert len(accepted['action_receipt_sha256']) == policy + 1
    checks.append('two_update_main_flow_preserves_native_HTTP400_body_and_accepts_actual_language_inequality')

proof = {'accepted': True, 'created': time.time(), 'checks': checks, 'count': len(checks),
         'invalid_prefix_cases': len(bad), 'actual_HTTP_GPU_or_model_calls': 0,
         'actual_live_chain_tested': False,
         'first_real_update_evidence': 'separate native v5 reconciliation proof'}
Path(__file__).with_name('CPU_ACCEPTED.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof, indent=2))
