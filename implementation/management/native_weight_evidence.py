"""Interpret the native checker's equality result and language-weight names."""

import json
import math
import re


def is_language_weight(name):
    if name.startswith('language_model.') and name.endswith('.weight'):
        return True
    if name in {'model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight'}:
        return True
    return re.fullmatch(r'model\.layers\.\d+\..+\.weight', name) is not None


def changed_language_tensors(response):
    if response.get('success') is not False:
        raise ValueError('Native checker did not report a tensor difference')
    message = response.get('message', '')
    if 'check tensor equality failed:' not in message:
        raise ValueError('Native checker failed without a tensor comparison')
    rows = []
    for name, maximum, mean in re.findall(
            r'name=(\S+) max_abs_err=(\S+) mean_abs_err=(\S+)', message):
        if not is_language_weight(name):
            continue
        maximum, mean = float(maximum), float(mean)
        if not math.isfinite(maximum) or not math.isfinite(mean):
            raise ValueError('Non-finite language-weight difference')
        if maximum <= 0 or mean <= 0:
            raise ValueError('Changed language weights require positive differences')
        rows.append({'name': name, 'max_abs_err': maximum, 'mean_abs_err': mean})
    if not rows:
        raise ValueError('No language-weight change was proven')
    return rows


def decode_weight_response(raw, status, action):
    response = json.loads(raw)
    if not isinstance(response, dict):
        raise ValueError('Native checker must return a JSON object')
    if status == 200 and response.get('success') is True:
        return response
    if action == 'compare' and status in {200, 400}:
        changed_language_tensors(response)
        return response
    raise ValueError('Native checker did not return a valid result for this action')
