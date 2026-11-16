"""No-network checks of the initial serving-snapshot lifecycle."""

from argparse import Namespace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from clm_replication.weight_provenance import capture_initial_snapshot


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / 'audit').mkdir()
        (self.root / 'attempt-08').mkdir()
        (self.root / 'CURRENT_OWNER.json').write_text(json.dumps({'attempt': 8, 'pid': 123}))
        self.args = Namespace(clm_audit_dir=str(self.root / 'audit'))

    def tearDown(self):
        self.temporary.cleanup()

    def responses(self, success=True, version='1'):
        return [io.BytesIO(json.dumps({'weight_version': version,
                'model_path': '/mount-data/Models/Qwen3.5-9B'}).encode()),
                io.BytesIO(json.dumps({'success': success}).encode())]

    def test_snapshot_after_sync_and_idempotent_success(self):
        with patch('urllib.request.urlopen', side_effect=self.responses()) as call:
            capture_initial_snapshot(self.args)
            self.assertEqual(call.call_count, 2)
            capture_initial_snapshot(self.args)
            self.assertEqual(call.call_count, 2)
        proof = json.loads((self.root / 'attempt-08/initial-weight-snapshot/ACCEPTED.json').read_text())
        self.assertTrue(proof['accepted'])
        self.assertEqual(proof['new_generation_calls'], 0)

    def test_unknown_attempt_is_not_resent(self):
        with patch('urllib.request.urlopen', side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                capture_initial_snapshot(self.args)
        with patch('urllib.request.urlopen') as call:
            with self.assertRaises(FileExistsError):
                capture_initial_snapshot(self.args)
            call.assert_not_called()

    def test_wrong_policy_is_rejected(self):
        with patch('urllib.request.urlopen', side_effect=self.responses(version='2')) as call:
            with self.assertRaises(AssertionError):
                capture_initial_snapshot(self.args)
            self.assertEqual(call.call_count, 1)

    def test_checker_rejection_is_retained(self):
        with patch('urllib.request.urlopen', side_effect=self.responses(success=False)):
            with self.assertRaises(AssertionError):
                capture_initial_snapshot(self.args)
        directory = self.root / 'attempt-08/initial-weight-snapshot'
        self.assertTrue((directory / 'response.raw').exists())
        self.assertFalse((directory / 'ACCEPTED.json').exists())


if __name__ == '__main__':
    unittest.main()
