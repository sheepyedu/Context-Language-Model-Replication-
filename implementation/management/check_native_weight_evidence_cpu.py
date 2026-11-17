"""Exercise native HTTP400 and physical language namespaces without requests."""

from copy import deepcopy
import json
from pathlib import Path
import time

from native_weight_evidence import changed_language_tensors, decode_weight_response
from first_update_reconcile_v5 import original_comparison


checks = []
for name in ['model.layers.0.q_proj.weight', 'model.embed_tokens.weight',
             'lm_head.weight', 'language_model.fixture.weight']:
    response = {'success': False, 'message': 'check tensor equality failed:\n'
                f'name={name} max_abs_err=0.000001 mean_abs_err=0.00000001'}
    assert changed_language_tensors(response)[0]['name'] == name
checks.append('physical_native_and_explicit_language_weight_names_preserved')
raw = json.dumps(response).encode()
assert decode_weight_response(raw, 400, 'compare') == response
checks.append('HTTP400_requires_actual_positive_finite_language_inequality')
assert decode_weight_response(raw, 200, 'compare') == response
checks.append('existing_200_inequality_fixture_contract_preserved')
assert decode_weight_response(b'{"success":true}', 200, 'snapshot')['success']
assert decode_weight_response(b'{"success":true}', 200, 'compare')['success']
checks.append('successful_snapshot_and_equal_comparison_remain_200')
bad = [
    (raw, 400, 'snapshot'), (raw, 500, 'compare'),
    (b'{"success":false,"message":"unrelated error"}', 400, 'compare'),
    (b'{"success":true}', 400, 'compare'),
    (b'{"success":false,"message":"check tensor equality failed: name=visual.proj.weight max_abs_err=1 mean_abs_err=.1"}', 400, 'compare'),
    (b'{"success":false,"message":"check tensor equality failed: name=model.rotary.inv_freq max_abs_err=1 mean_abs_err=.1"}', 400, 'compare'),
]
for candidate, status, action in bad:
    try:
        decode_weight_response(candidate, status, action)
    except ValueError:
        pass
    else:
        raise AssertionError((candidate, status, action))
checks.append('unrelated_errors_visual_changes_and_buffers_do_not_prove_language_parameter_changes')
for maximum, mean in [('nan', '.1'), ('inf', '.1'), ('0', '.1'), ('1', '0'), ('-1', '.1')]:
    candidate = {'success': False, 'message': 'check tensor equality failed:\n'
                 f'name=lm_head.weight max_abs_err={maximum} mean_abs_err={mean}'}
    try:
        changed_language_tensors(candidate)
    except ValueError:
        pass
    else:
        raise AssertionError(candidate)
checks.append('original_finite_and_positive_difference_thresholds_preserved')
base = Path(__file__).resolve().parent
intent = json.loads((base / 'first-update-acceptance-v4/INTENT.json').read_text())
original = (base / 'first-update-reconciliation-v5/ORIGINAL_COMPARE_SERVER.log').read_bytes()
changes = original_comparison(original, intent)
assert any(row['name'] == 'lm_head.weight' for row in changes)
assert any(row['name'].startswith('model.layers.') for row in changes)
checks.append('original_saved_four_rank_comparison_proves_language_head_and_layer_weight_changes')
wrong = deepcopy(intent)
wrong['created'] += 60
for candidate, candidate_intent in [
    (original, wrong),
    (original.replace(b'TP3] [WeightChecker] handle action=compare', b'TP2] [WeightChecker] handle action=compare'), intent),
    (original.replace(b'400 Bad Request', b'200 OK'), intent),
    (original + b'\n[WeightChecker] handle action=snapshot\n', intent),
]:
    try:
        original_comparison(candidate, candidate_intent)
    except ValueError:
        pass
    else:
        raise AssertionError('Altered original comparison was accepted')
checks.append('wrong_intent_time_rank_completion_or_intervening_snapshot_rejected')
proof = {'accepted': True, 'created': time.time(), 'count': len(checks), 'checks': checks,
         'actual_original_language_weight_records': len(changes),
         'actual_original_unique_language_weight_names': len({row['name'] for row in changes}),
         'actual_HTTP_GPU_or_model_calls': 0, 'original_client_HTTP_body_missing_preserved': True}
(base / 'NATIVE_WEIGHT_EVIDENCE_CPU_ACCEPTED.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof, indent=2))
