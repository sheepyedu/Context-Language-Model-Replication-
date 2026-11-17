"""Loopback-only public web retrieval for the PAI task containers."""

from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import gzip
import hashlib
import json
from pathlib import Path
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent / 'web-retrieval'
ROOT.mkdir(exist_ok=True)


def validate_url(url):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password:
        raise ValueError('Public HTTP(S) URLs are required')
    if parts.port not in (None, 80, 443):
        raise ValueError('Only public web ports are supported')
    for address in socket.getaddrinfo(parts.hostname, parts.port or 443, type=socket.SOCK_STREAM):
        ip = ipaddress.ip_address(address[4][0])
        clash_fake = ip.version == 4 and ip in ipaddress.ip_network('198.18.0.0/15')
        if not ip.is_global and not clash_fake:
            raise ValueError('Private addresses cannot be retrieved')


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        validate_url(url)
        return super().redirect_request(request, response, code, message, headers, url)


class ReadableHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attributes):
        if tag in ('script', 'style', 'noscript'):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript') and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def retrieve(payload):
    action = payload['action']
    if action == 'search':
        url = 'https://www.bing.com/search?format=rss&q=' + urllib.parse.quote(payload['query'])
    elif action == 'fetch':
        url = payload['url']
    else:
        raise ValueError('Unknown retrieval action')
    validate_url(url)
    opener = urllib.request.build_opener(SafeRedirect())
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    started = time.time()
    with opener.open(request, timeout=25) as response:
        raw = response.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError('Document exceeds the 8 MiB retrieval limit')
        text = raw.decode(response.headers.get_content_charset() or 'utf-8', errors='replace')
        final_url = response.geturl()
    if action == 'search':
        document = ET.fromstring(text)
        result = {'results': [
            {'title': item.findtext('title'), 'url': item.findtext('link'),
             'snippet': (item.findtext('description') or '')[:2048]}
            for item in document.findall('.//item')[:10]
        ]}
    else:
        parser = ReadableHTML()
        parser.feed(text)
        content = '\n'.join(parser.parts) if parser.parts else text
        result = {'url': final_url, 'text': content[:32768], 'truncated': len(content) > 32768}
    identifier = uuid.uuid4().hex
    directory = ROOT / identifier
    directory.mkdir()
    (directory / 'body.raw.gz').write_bytes(gzip.compress(raw))
    (directory / 'receipt.json').write_text(json.dumps({
        'request': payload, 'url': final_url, 'started': started, 'finished': time.time(),
        'body_sha256': hashlib.sha256(raw).hexdigest(), 'result': result,
    }, ensure_ascii=False, indent=2))
    result['retrieval_id'] = identifier
    return result


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16384:
                raise ValueError('Invalid request size')
            payload = json.loads(self.rfile.read(length))
            result = retrieve(payload)
            code = 200
        except Exception as error:
            result = {'error': type(error).__name__, 'message': str(error)}
            code = 502
        body = json.dumps(result, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', 19180), Handler).serve_forever()
