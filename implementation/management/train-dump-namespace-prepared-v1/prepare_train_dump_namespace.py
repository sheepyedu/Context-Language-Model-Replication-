"""Prepare rank-separated diagnostic filenames for a future owned launch."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import string


ARGUMENT = re.compile(
    r'(?P<prefix>--save-debug-train-data[ \t]+)'
    r'(?P<value>"[^"\n]*"|\'[^\'\n]*\'|[^\s\\]+)'
)


def prepare(source):
    matches = list(ARGUMENT.finditer(source))
    if len(matches) != 1:
        raise ValueError('Expected one native debug-train filename argument')
    match = matches[0]
    token = match.group('value')
    values = shlex.split(token)
    if len(values) != 1:
        raise ValueError('Debug-train filename is not one shell argument')
    template = values[0]
    parts = list(string.Formatter().parse(template))
    fields = [field for _, field, _, _ in parts if field is not None]
    if 'rollout_id' not in fields or set(fields) - {'rank', 'rollout_id'}:
        raise ValueError('Debug filename needs native rollout_id and optional rank fields')
    if any(spec or conversion for _, _, spec, conversion in parts):
        raise ValueError('Unexpected filename formatting conversion')
    if 'rank' in fields:
        return source, template, template
    stem, extension = os.path.splitext(template)
    replacement = stem + '-rank-{rank}' + extension
    if token.startswith(('"', "'")):
        replacement_token = token[0] + replacement + token[0]
    else:
        replacement_token = replacement
    updated = source[:match.start('value')] + replacement_token + source[match.end('value'):]
    assert updated[:match.start('value')] == source[:match.start('value')]
    assert updated[match.start('value') + len(replacement_token):] == source[match.end('value'):]
    return updated, template, replacement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expected-input-sha256', required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError('The original launch file must be preserved')
    original = args.input.read_bytes()
    input_sha = hashlib.sha256(original).hexdigest()
    if input_sha != args.expected_input_sha256:
        raise ValueError('Input launcher differs from the explicitly selected source')
    updated, old_template, new_template = prepare(original.decode())
    payload = updated.encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        handle.write(payload)
    args.output.chmod(0o700)
    record = {'prepared': True, 'input': str(args.input), 'output': str(args.output),
              'input_sha256': input_sha, 'output_sha256': hashlib.sha256(payload).hexdigest(),
              'old_template': old_template, 'new_template': new_template,
              'scope': 'diagnostic output filenames only; apply at a future owned launch',
              'launch_performed': False}
    args.output.with_name(args.output.name + '.receipt.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


if __name__ == '__main__':
    main()
