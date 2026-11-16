"""Run the actual web CLI through Polar's route/registry and the real corpus."""
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import urllib.error
import urllib.request

R=Path('/mount-data/CLM-Replication')
P=R/'management/retrieval-gateway-prepared-v2'
for path in [R/'management/retrieval-v1',R/'management/fixed-corpus-api-prepared',R/'src/ProRL/src',P/'source']:
    sys.path.insert(0,str(path))
import uvicorn
import polar.gateway.server as native_server
from polar.gateway.session import SessionRegistry, SessionStatus
from clm_replication import gateway, retrieval_routing
from fixed_corpus import FixedCorpus
from fixed_corpus_api import create_server
assert Path(gateway.__file__)==P/'source/clm_replication/gateway.py'
assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
assert not getattr(sys.modules.get('torch'),'cuda',SimpleNamespace(is_initialized=lambda:False)).is_initialized()
directory=P/f'http-{time.time_ns()}'
directory.mkdir()
(directory/'INTENT.json').write_text(json.dumps({'created':time.time(),'pid':os.getpid(),
    'operation':'finite native gateway/web-CLI/corpus integration, zero model requests'}))
asset=json.loads((R/'management/RETRIEVAL_ASSETS_ACCEPTED.json').read_text())
corpus=FixedCorpus(asset['index_path'],'/mount-data/Models/Qwen3.5-9B')
corpus_server=create_server(corpus,directory/'corpus-records',0)

class PublicHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        raw=self.rfile.read(int(self.headers.get('Content-Length','0')))
        self.server.requests.append(json.loads(raw))
        body=json.dumps({'public_web':True,'received':json.loads(raw)}).encode()
        self.send_response(200)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers();self.wfile.write(body)
    def log_message(self,*args):
        pass

public_server=HTTPServer(('127.0.0.1',0),PublicHandler)
public_server.requests=[]
retrieval_routing.BACKENDS.update({
    'public_web':f'http://127.0.0.1:{public_server.server_port}',
    'browsecomp_plus_fixed':f'http://127.0.0.1:{corpus_server.server_port}/retrieve'})
registry=SessionRegistry()
private='driver-private-label-fixture-not-a-benchmark-answer'
fixed={'dataset':'browsecomp_plus','retrieval_profile':'browsecomp_plus_fixed','private_label':private}
corpus_sid=registry.register('sk-polar-corpus-fixture',task_id='cpu-fixture',metadata=fixed).session_id
public_sid=registry.register('sk-polar-public-fixture',metadata={'private_label':private}).session_id
missing_sid=registry.register('sk-polar-missing-profile',metadata={'dataset':'browsecomp_plus'}).session_id
unknown_sid=registry.register('sk-polar-unknown-profile',metadata={'retrieval_profile':'unprovided'}).session_id
terminal_sid=registry.register('sk-polar-terminal-fixture',status=next(iter(SessionStatus.terminal())),metadata=fixed).session_id
model_access=[]
class RejectInference:
    def __getattr__(self,name):
        model_access.append(name)
        raise AssertionError('This finite CPU check must not call a model')
native_server._state=SimpleNamespace(session_registry=registry,inference=RejectInference())
assert gateway.app.router.routes[0].path=='/clm/web'
server=uvicorn.Server(uvicorn.Config(gateway.app,host='127.0.0.1',port=0,
    lifespan='off',log_level='error'))
threads=[threading.Thread(target=corpus_server.serve_forever,kwargs={'poll_interval':.1},daemon=True),
         threading.Thread(target=public_server.serve_forever,kwargs={'poll_interval':.1},daemon=True),
         threading.Thread(target=server.run,daemon=True)]
for thread in threads:thread.start()
checks=[]
try:
    deadline=time.monotonic()+20
    while not server.started:
        if time.monotonic()>deadline:raise TimeoutError('Temporary native gateway startup timed out')
        time.sleep(.05)
    port=server.servers[0].sockets[0].getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    (directory/'SERVERS_STARTED.json').write_text(json.dumps({'pid':os.getpid(),
        'gateway_port':port,'corpus_port':corpus_server.server_port,
        'public_fixture_port':public_server.server_port,'created':time.time()},indent=2))
    def request(sid,payload):
        raw=json.dumps(payload).encode()
        req=urllib.request.Request(base+'/clm/web',data=raw,
            headers={'Content-Type':'application/json','Authorization':'Bearer '+sid})
        try:
            with urllib.request.urlopen(req,timeout=30) as response:
                return response.status,json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code,json.loads(error.read())
    def cli(sid,action,value):
        env=dict(os.environ,OPENAI_BASE_URL=base+'/v1',SESSION_ID=sid,
                 PYTHONPATH=str(P/'source')+os.pathsep+os.environ.get('PYTHONPATH',''))
        command=[sys.executable,'-m','clm_replication.web',action,value]
        started=time.time()
        result=subprocess.run(command,env=env,capture_output=True,timeout=35)
        root=directory/f'cli-{len(list(directory.glob("cli-*"))):02d}'
        root.mkdir();(root/'stdout.raw').write_bytes(result.stdout);(root/'stderr.raw').write_bytes(result.stderr)
        (root/'RECEIPT.json').write_text(json.dumps({'command':command,'started':started,
            'finished':time.time(),'integer_exit':result.returncode,
            'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),
            'stderr_sha256':hashlib.sha256(result.stderr).hexdigest()},indent=2))
        assert result.returncode==0,result.stderr.decode(errors='replace')
        return json.loads(result.stdout)
    search=cli(corpus_sid,'search','Paris France')
    assert len(search['results'])==10
    checks.append('native_cli_to_real_fixed_corpus_search')
    assert all(row['snippet_tokens']<=512 for row in search['results'])
    checks.append('all_ten_result_snippets_within_512_tokens')
    document=cli(corpus_sid,'fetch',search['results'][0]['url'])
    assert document['document_tokens']<=8192
    checks.append('native_cli_fetch_within_8192_tokens')
    assert corpus.raw_document(document['docid']).startswith(document['text'])
    checks.append('fetched_text_is_exact_source_prefix')
    public=cli(public_sid,'search','generic fixture query')
    assert public=={'public_web':True,'received':{'action':'search','query':'generic fixture query'}}
    checks.append('native_public_web_cli_contract_preserved')
    payload={'action':'search','query':'generic fixture query','legacy_extra':'kept'}
    status,value=request(public_sid,payload)
    assert status==200 and value['received']==payload
    checks.append('public_web_extra_fields_preserved')
    for sid,label in [('sk-polar-unregistered-fixture','unregistered_session_rejected'),
                      (terminal_sid,'terminal_native_session_rejected')]:
        assert request(sid,{'action':'search','query':'Paris France'})[0]==403
        checks.append(label)
    for sid,label in [(missing_sid,'fixed_dataset_requires_driver_profile'),
                      (unknown_sid,'unknown_driver_profile_rejected')]:
        assert request(sid,{'action':'search','query':'Paris France'})[0]==400
        checks.append(label)
    for extra,label in [({'retrieval_profile':'public_web'},'model_cannot_override_driver_profile'),
                        ({'label':private},'fixed_payload_private_label_rejected')]:
        assert request(corpus_sid,{'action':'search','query':'Paris France',**extra})[0]==400
        checks.append(label)
    assert request(corpus_sid,{'action':'fetch','url':'https://example.org'})[0]==400
    checks.append('fixed_fetch_rejects_external_url')
    assert request(corpus_sid,{'action':'fetch','url':'bcp://unprovided-document-id'})[0]==404
    checks.append('fixed_missing_document_404_relayed')
    assert len(public_server.requests)==2
    assert all(private not in json.dumps(row) for row in public_server.requests)
    checks.append('private_driver_metadata_not_sent_to_public_backend')
finally:
    server.should_exit=True
    corpus_server.shutdown();public_server.shutdown()
    corpus_server.server_close();public_server.server_close()
    for thread in threads:thread.join(timeout=10)
assert not any(thread.is_alive() for thread in threads)
checks.append('all_three_temporary_http_servers_closed')
receipts=list((directory/'corpus-records').glob('*/RECEIPT.json'))
assert len(receipts)==4,len(receipts)
for path in receipts:
    value=json.loads(path.read_text());parent=path.parent
    raw=(parent/'request.raw').read_bytes()
    assert private.encode() not in raw
    assert hashlib.sha256(raw).hexdigest()==value['request_sha256']
    assert hashlib.sha256((parent/'response.raw').read_bytes()).hexdigest()==value['response_sha256']
checks.append('all_four_backend_raw_request_response_hashes_verified')
assert not model_access
checks.append('native_inference_backend_access_zero')
proof={'accepted':True,'created':time.time(),'checks':checks,'count':len(checks),
    'native_gateway_source':str(Path(gateway.__file__)),'records':str(directory),
    'actual_native_cli_processes':3,'backend_raw_receipts':4,'model_calls':0,
    'benchmark_question_or_answer_used':False,'GPU_used':False,
    'active_C2_gateway_changed':False,'production_activation':False,
    'scope':'native web CLI, actual Polar ASGI route and SessionRegistry, real corpus HTTP; actual model benchmark task pending'}
(directory/'ACCEPTED.json').write_text(json.dumps(proof,indent=2))
(P/'HTTP_NATIVE_ACCEPTED.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof,indent=2))
