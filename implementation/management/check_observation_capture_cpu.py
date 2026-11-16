"""Regress the observed read-before-SSH-completion race without networking."""
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

import capture_training_observation as module

checks = []
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    latest = root / 'OBSERVATION_LATEST.json'
    previous = b'{"previous_complete": true}'
    latest.write_bytes(previous)
    observation = {'observed': 1, 'owner': {}, 'actual_owner': {}, 'native_progress': []}
    raw = json.dumps(observation).encode()

    def completed(*args, **kwargs):
        assert latest.read_bytes() == previous
        assert kwargs['timeout'] == 45
        return subprocess.CompletedProcess(args[0], 0, raw, b'')

    with patch.object(module.subprocess, 'run', side_effect=completed):
        assert module.capture(root) == observation
    assert latest.read_bytes() == raw
    checks.append('complete_JSON_published_after_transport_finishes')

    failures = [
        (subprocess.CompletedProcess([], 0, b'{"partial":', b''), json.JSONDecodeError),
        (subprocess.CompletedProcess([], 255, b'', b'fixture connection error'), subprocess.CalledProcessError),
        (subprocess.TimeoutExpired([], 45, output=b'{"partial":', stderr=b'timed out'), subprocess.TimeoutExpired),
    ]
    for result, error_type in failures:
        with patch.object(module.subprocess, 'run') as runner:
            if isinstance(result, Exception):
                runner.side_effect = result
            else:
                runner.return_value = result
            try:
                module.capture(root)
            except error_type:
                pass
            else:
                raise AssertionError(error_type)
        assert latest.read_bytes() == raw
        checks.append(error_type.__name__ + '_preserves_previous_complete_observation')
    assert len(list((root / 'observations').glob('*.json'))) == 4
print(json.dumps({'accepted': True, 'checks': checks, 'count': len(checks),
                  'actual_network_or_model_calls': 0}, indent=2))
