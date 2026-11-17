"""Teacher-force saved engineering tokens after the first actor weight sync."""

import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request
import uuid

from clm_replication.accounting import record_tokens, summarize

ROOT = Path('/mount-data/CLM-Replication')
source = ROOT / 'management/functional-trajectory-02/RESULT.json'
output = ROOT / 'management/conversion-logprob-parity'
output.mkdir(exist_ok=True)
traces = json.loads(source.read_text())['results'][0]['trajectory']['traces']
rows = []
for index, trace in enumerate(traces):
    directory = output / f'call-{index:02d}'
    directory.mkdir(exist_ok=True)
    if (directory / 'RESULT.json').exists():
        rows.append(json.loads((directory / 'RESULT.json').read_text()))
        continue
    if (directory / 'INTENT.json').exists():
        raise RuntimeError('Existing parity request requires reconciliation before continuation')
    tokens = trace['prompt_ids'] + trace['response_ids']
    payload = {'input_ids': tokens, 'sampling_params': {'max_new_tokens': 0, 'temperature': 0.7,
               'top_p': 0.95}, 'return_logprob': True, 'logprob_start_len': 0}
    request = json.dumps(payload).encode()
    call_id = uuid.uuid4().hex
    with (directory / 'INTENT.json').open('x') as handle:
        json.dump({'call_id': call_id, 'created': time.time(), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                   'source_call': index, 'request_sha256': hashlib.sha256(request).hexdigest()}, handle)
    (directory / 'request.json').write_bytes(request)
    try:
        response = urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:19174/generate',
                                          data=request, headers={'Content-Type': 'application/json'}), timeout=180)
        raw = response.read()
        (directory / 'response.raw').write_bytes(raw)
        parsed = json.loads(raw)
    except Exception as error:
        (directory / 'ERROR.json').write_text(json.dumps({'type': type(error).__name__, 'message': str(error)}))
        record_tokens(directory, call_id=call_id, role='conversion_validation', model_requested='qwen35-clm',
                      scope={'purpose': 'engineering_conversion_logprob'}, error={'type': type(error).__name__})
        raise
    meta = parsed['meta_info']
    record_tokens(directory, call_id=call_id, role='conversion_validation', model_requested='qwen35-clm',
                  response={'usage': {'prompt_tokens': meta.get('prompt_tokens'),
                                     'completion_tokens': meta.get('completion_tokens')}},
                  scope={'purpose': 'engineering_conversion_logprob'})
    values = meta['input_token_logprobs']
    assert len(values) == len(tokens), (len(values), len(tokens))
    assert [value[1] for value in values] == tokens, 'Teacher-forcing token alignment differs'
    probabilities = [value[0] for value in values[len(trace['prompt_ids']):]]
    original = trace['response_logprobs']
    assert all(isinstance(value, (float, int)) and math.isfinite(value) for value in probabilities)
    differences = [abs(first-second) for first, second in zip(probabilities, original)]
    assert len(probabilities) == len(original)
    row = {'call': index, 'tokens': len(original), 'max_absolute_logprob_difference': max(differences),
           'mean_absolute_logprob_difference': sum(differences)/len(differences),
           'accepted': max(differences) <= 0.1 and sum(differences)/len(differences) <= 0.01,
           'tolerance': {'max_absolute': 0.1, 'mean_absolute': 0.01},
           'note': 'BF16 prefill versus sampled decode can differ. All original values and deviations remain available.'}
    (directory / 'RESULT.json').write_text(json.dumps(row, indent=2))
    rows.append(row)
result = {'source': str(source), 'condition': 'initial actor converted checkpoint after weight synchronization, before optimizer',
          'calls': rows, 'accepted': len(rows) == 4 and all(row['accepted'] for row in rows),
          'task_or_tool_replay': 0, 'new_generated_tokens_requested': 0}
(output / 'RESULT.json').write_text(json.dumps(result, indent=2))
summarize(ROOT)
print(json.dumps(result, indent=2))
