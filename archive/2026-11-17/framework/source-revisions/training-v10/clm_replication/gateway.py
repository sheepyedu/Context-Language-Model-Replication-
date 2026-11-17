"""Run Polar with durable task artifacts before its temporary-dir cleanup."""

import argparse
import os
from contextlib import asynccontextmanager
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import contextvars
import time
import uuid

from polar.gateway.node import GatewayNodeManager
from polar.gateway.server import app, get_state, serve
from fastapi import Header, HTTPException
import httpx
from polar.gateway.proxy import InferenceClient
from .accounting import record_tokens


ROOT = Path('/mount-data/CLM-Replication')

if os.environ.get('CLM_GATEWAY_START_PAUSED') == '1':
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def paused_lifespan(application):
        async with original_lifespan(application):
            await get_state().inference.pause_generation(timeout_seconds=1)
            yield

    app.router.lifespan_context = paused_lifespan
original_cleanup = GatewayNodeManager._remove_session_dir_best_effort
original_completion = InferenceClient.completion
request_session = contextvars.ContextVar('clm_request_session', default='unattributed')


@app.middleware('http')
async def bind_request_session(request, call_next):
    session_id = request.headers.get('authorization', '').removeprefix('Bearer ')
    token = request_session.set(session_id if session_id.startswith('sk-polar-') else 'unattributed')
    try:
        return await call_next(request)
    finally:
        request_session.reset(token)


async def recorded_completion(self, request):
    directory = ROOT / 'runs/provider-attempts' / request_session.get() / uuid.uuid4().hex
    directory.mkdir(parents=True)
    info = get_state().session_registry.get(request_session.get())
    scope = {'session_id': request_session.get(), 'task_id': info.task_id if info else None,
             **(info.metadata if info else {})}
    (directory / 'INTENT.json').write_text(json.dumps({'created': time.time(), 'request': request}))
    try:
        response = await original_completion(self, request)
    except Exception as error:
        (directory / 'ERROR.json').write_text(json.dumps({
            'finished': time.time(), 'type': type(error).__name__, 'message': str(error),
            'body': getattr(error, 'body', None),
        }, default=str))
        record_tokens(directory, call_id=directory.name, role='rollout',
                      model_requested=request.get('model'), scope=scope,
                      error={'type': type(error).__name__, 'message': str(error)})
        raise
    (directory / 'RESPONSE.json').write_text(json.dumps({'finished': time.time(), 'response': response}))
    record_tokens(directory, call_id=directory.name, role='rollout',
                  model_requested=request.get('model'), response=response, scope=scope)
    return response


@app.post('/clm/web')
async def public_web(payload: dict, authorization: str = Header(default='')):
    session_id = authorization.removeprefix('Bearer ')
    if session_id not in {item['session_id'] for item in get_state().session_registry.active_sessions()}:
        raise HTTPException(status_code=403, detail='An active task session is required')
    async with httpx.AsyncClient(timeout=45, trust_env=False) as client:
        response = await client.post('http://127.0.0.1:19180', json=payload)
        response.raise_for_status()
        return response.json()


# Polar's POST catch-all is registered when its server module is imported.
# The explicit tool route must be matched before that proxy route.
web_route = next(route for route in app.router.routes if getattr(route, 'path', None) == '/clm/web')
app.router.routes.remove(web_route)
app.router.routes.insert(0, web_route)


async def archive_then_cleanup(self, session_dir, session_id):
    destination = ROOT / 'runs/session-artifacts' / session_id
    if not session_dir.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)

    def archive():
        if destination.exists():
            raise FileExistsError(f'Existing archive must be reconciled: {destination}')
        temporary = destination.with_name(destination.name + '.partial')
        shutil.copytree(session_dir, temporary)
        files = {}
        for path in sorted(temporary.rglob('*')):
            if path.is_file():
                files[str(path.relative_to(temporary))] = hashlib.sha256(path.read_bytes()).hexdigest()
        (temporary / 'ARCHIVE_MANIFEST.json').write_text(json.dumps(files, indent=2))
        temporary.rename(destination)

    # A failed archive leaves the original directory available for recovery.
    await asyncio.to_thread(archive)
    await original_cleanup(self, session_dir, session_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topology', required=True)
    parser.add_argument('--node-id', required=True)
    args = parser.parse_args()
    GatewayNodeManager._remove_session_dir_best_effort = archive_then_cleanup
    InferenceClient.completion = recorded_completion
    serve(args.topology, node_id=args.node_id)


if __name__ == '__main__':
    main()
