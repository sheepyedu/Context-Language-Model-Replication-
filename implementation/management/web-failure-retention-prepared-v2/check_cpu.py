"""Compare the original and prepared HTTP handlers using local CPU fixtures."""

from contextlib import redirect_stderr
import hashlib
import http.client
import importlib.util
import io
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest import mock
import urllib.error


HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / 'original_broker.py'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def request_handler(module, body, error=None, result=None, archive_failure=False):
    with tempfile.TemporaryDirectory() as temporary:
        module.ROOT = Path(temporary)
        server = module.ThreadingHTTPServer(('127.0.0.1', 0), module.Handler)
        worker = threading.Thread(target=server.handle_request)
        retrieve = mock.Mock(side_effect=error, return_value=result)
        archive_mock = mock.Mock(side_effect=OSError('fixture storage failure'))
        stderr = io.StringIO()
        patches = [mock.patch.object(module, 'retrieve', retrieve)]
        if archive_failure:
            patches.append(mock.patch.object(module.Path, 'mkdir', archive_mock))
        try:
            with redirect_stderr(stderr):
                for patch in patches:
                    patch.start()
                worker.start()
                connection = http.client.HTTPConnection(*server.server_address, timeout=5)
                connection.request('POST', '/', body=body)
                response = connection.getresponse()
                actual = (response.status, response.read(), response.getheader('Content-Type'))
                connection.close()
                worker.join(timeout=5)
                if worker.is_alive():
                    raise AssertionError('fixture HTTP worker did not finish')
        finally:
            for patch in reversed(patches):
                patch.stop()
            server.server_close()
        receipts = list(Path(temporary).glob('failure-*/receipt.json'))
        archives = []
        for receipt in receipts:
            directory = receipt.parent
            archives.append({
                'receipt': json.loads(receipt.read_text()),
                'request': (directory / 'request.raw').read_bytes(),
                'response': (directory / 'error-response.raw').read_bytes(),
                'traceback': (directory / 'traceback.txt').read_text(),
            })
        return actual, archives, retrieve.call_count, stderr.getvalue()


class RetentionCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = load_module(ORIGINAL, 'original_broker_fixture')
        cls.candidate = load_module(HERE / 'public_web_broker.py', 'prepared_broker_fixture')

    def assert_failure(self, body, error=None):
        old = request_handler(self.original, body, error=error)
        new = request_handler(self.candidate, body, error=error)
        self.assertEqual(old[0], new[0])
        self.assertEqual(new[0][0], 502)
        self.assertEqual(new[2], old[2])
        self.assertEqual(len(new[1]), 1)
        archive = new[1][0]
        self.assertEqual(archive['request'], body)
        self.assertEqual(archive['response'], new[0][1])
        self.assertEqual(archive['receipt']['request_sha256'], hashlib.sha256(body).hexdigest())
        self.assertIn('Traceback', archive['traceback'])
        self.assertFalse(archive['receipt']['upstream_body_retained'])
        return archive

    def test_http403(self):
        error = urllib.error.HTTPError('https://fixture.invalid/', 403, 'Forbidden', {}, None)
        archive = self.assert_failure(b'{"action":"fetch","url":"https://fixture.invalid/"}', error)
        self.assertEqual(archive['receipt']['upstream_http_status'], 403)

    def test_dns_error(self):
        self.assert_failure(b'{"action":"search","query":"fixture"}', urllib.error.URLError('fixture DNS'))

    def test_timeout(self):
        self.assert_failure(b'{"action":"search","query":"fixture"}', socket.timeout('fixture timeout'))

    def test_invalid_json(self):
        self.assert_failure(b'{invalid')

    def test_unicode(self):
        self.assert_failure('{"action":"search","query":"测试"}'.encode(), ValueError('测试失败'))

    def test_archive_failure_preserves_response(self):
        body = b'{"action":"search","query":"fixture"}'
        error = ValueError('fixture retrieval failed')
        old = request_handler(self.original, body, error=error)
        new = request_handler(self.candidate, body, error=error, archive_failure=True)
        self.assertEqual(old[0], new[0])
        self.assertIn('Retrieval error archive failed', new[3])
        self.assertEqual(new[1], [])

    def test_success_unchanged(self):
        body = b'{"action":"search","query":"fixture"}'
        result = {'results': [{'title': 'fixture'}], 'retrieval_id': 'fixture-id'}
        old = request_handler(self.original, body, result=result)
        new = request_handler(self.candidate, body, result=result)
        self.assertEqual(old[0], new[0])
        self.assertEqual(new[0][0], 200)
        self.assertEqual(new[1], [])
        self.assertEqual(new[2], 1)

    def test_empty_request(self):
        archive = self.assert_failure(b'')
        self.assertFalse(archive['receipt']['request_body_complete'])

    def test_oversized_request_is_not_read(self):
        body = b'x' * 16385
        old = request_handler(self.original, body)
        new = request_handler(self.candidate, body)
        self.assertEqual(old[0], new[0])
        self.assertEqual(new[1][0]['request'], b'')
        self.assertEqual(new[1][0]['receipt']['declared_content_length'], len(body))
        self.assertFalse(new[1][0]['receipt']['request_body_complete'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
