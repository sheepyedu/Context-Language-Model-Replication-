"""Read native optimizer metrics and compare the current serving weights once."""

import ast
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
import urllib.request


ROOT = Path('/mount-data/CLM-Replication')
DIRECTORY = ROOT / 'management/first-update-acceptance-v2'


def changed_language_tensors(response):
    """An equality-check failure must contain actual finite language changes."""
    if response.get('success') is not False:
        raise ValueError('Native checker did not report a tensor difference')
    message = response.get('message', '')
    if 'check tensor equality failed:' not in message:
        raise ValueError('Native checker failed without a tensor comparison')
    rows = []
    for name, maximum, mean in re.findall(
        r'name=(\S+) max_abs_err=(\S+) mean_abs_err=(\S+)', message
    ):
        if 'language_model.' not in name:
            continue
        maximum, mean = float(maximum), float(mean)
        if not math.isfinite(maximum) or not math.isfinite(mean):
            raise ValueError('Non-finite parameter difference')
        if maximum <= 0 or mean <= 0:
            raise ValueError('Changed tensor must have a positive difference')
        rows.append({'name': name, 'max_abs_err': maximum, 'mean_abs_err': mean})
    if not rows:
        raise ValueError('No language tensor change was proven')
    return rows


def native_actor_metrics(path):
    metrics = []
    for line in path.open(errors='replace'):
        match = re.search(r'(?<!critic-)step (\d+): (\{.*\})', line)
        if match and 'train/grad_norm' in line and 'train/step' in line:
            value = ast.literal_eval(match.group(2).split('\x1b')[0])
            gradient = float(value['train/grad_norm'])
            if not math.isfinite(gradient) or gradient <= 0:
                raise ValueError('Real native gradient must be finite and positive')
            losses = {key: item for key, item in value.items()
                      if 'loss' in key and isinstance(item, (float, int))}
            if not losses or not all(math.isfinite(item) for item in losses.values()):
                raise ValueError('Real native losses must be finite')
            metrics.append(value)
    return metrics


def get_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--owner-start', type=int, required=True)
    parameters = parser.parse_args()
    run = ROOT / 'runs/c2-seed1234'
    owner = json.loads((run / 'CURRENT_OWNER.json').read_bytes())
    assert owner['attempt'] == parameters.attempt
    proc = Path('/proc') / str(owner['pid'])
    assert proc.exists()
    assert int((proc / 'stat').read_text().split()[21]) == parameters.owner_start
    assert b'owned_training_v8.py' in (proc / 'cmdline').read_bytes()
    metrics = native_actor_metrics(Path(owner['directory']) / 'controller.log')
    assert metrics, 'No real optimizer update has happened yet'
    model = get_json('http://10.10.15.17:15000/model_info')
    assert model['weight_version'] == '2', model
    assert model['model_path'] == '/mount-data/Models/Qwen3.5-9B'
    snapshot = Path(owner['directory']) / 'initial-weight-snapshot'
    assert json.loads((snapshot / 'response.raw').read_bytes())['success']
    DIRECTORY.mkdir(exist_ok=True)
    with (DIRECTORY / 'INTENT.json').open('x') as handle:
        json.dump({'created': time.time(), 'attempt': parameters.attempt, 'logical_policy': 1,
                   'source_snapshot_sha256': hashlib.sha256((snapshot / 'response.raw').read_bytes()).hexdigest(),
                   'model_before': model, 'operation': 'native_read_only_weights_checker_compare'}, handle, indent=2)
    payload = json.dumps({'action': 'compare'}).encode()
    (DIRECTORY / 'request.json').write_bytes(payload)
    request = urllib.request.Request('http://10.10.15.17:15000/weights_checker',
                                    data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        raw = response.read()
    (DIRECTORY / 'response.raw').write_bytes(raw)
    changes = changed_language_tensors(json.loads(raw))
    after = get_json('http://10.10.15.17:15000/model_info')
    assert after == model
    proof = {'accepted': True, 'created': time.time(), 'attempt': parameters.attempt,
             'scope': 'first native actor update, finite losses/gradient and synchronized language-weight change',
             'native_metrics': metrics, 'logical_policy': 1, 'raw_serving_version': '2',
             'changed_language_tensors': changes,
             'response_sha256': hashlib.sha256(raw).hexdigest(),
             'new_generation_calls': 0, 'checkpoint_restore_accepted': False,
             'T06_complete': False}
    (DIRECTORY / 'ACCEPTED.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps({key: value for key, value in proof.items()
                      if key not in ('changed_language_tensors', 'native_metrics')}, indent=2))


if __name__ == '__main__':
    main()
