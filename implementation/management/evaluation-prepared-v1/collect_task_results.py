"""Bounded, resumable read-only export of original coordinator task responses."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import time
import urllib.error
import urllib.parse
import urllib.request

from question_metrics import aggregate_questions


def collect(manifest_raw, directory, base_url, seconds=35, opener=urllib.request.urlopen):
    if not 0 < seconds <= 35:
        raise ValueError('One export batch must be bounded to at most 35 seconds')
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / 'EXPORT.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return collect_locked(manifest_raw, directory, base_url, seconds, opener)


def collect_locked(manifest_raw, directory, base_url, seconds, opener):
    manifest = json.loads(manifest_raw)
    aggregate_questions(manifest, {})
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    latest = directory / 'TASK_RESULTS_INDEX.json'
    if latest.exists():
        index = json.loads(latest.read_text())
        if index['manifest_sha256'] != manifest_sha:
            raise ValueError('Export directory belongs to a different frozen evaluation')
    else:
        index = {'manifest_sha256': manifest_sha, 'responses': {}}
    responses = index['responses']
    raw_directory = directory / 'raw'
    raw_directory.mkdir(exist_ok=True, mode=0o700)
    errors = []
    reads = 0
    deadline = time.monotonic() + seconds
    previous_handler = signal.getsignal(signal.SIGALRM)
    def expire(signum, frame):
        raise TimeoutError('Bounded task-result export expired')
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        for case in manifest['cases']:
            task_id = case['task_id']
            saved = responses.get(task_id)
            if saved and saved['terminal_task'] and Path(saved['path']).exists():
                continue
            prefix = raw_directory / (hashlib.sha256(task_id.encode()).hexdigest() + '-' + str(time.time_ns()))
            raw_path = prefix.with_suffix('.raw')
            receipt_path = prefix.with_suffix('.json')
            started = time.time()
            digest = hashlib.sha256()
            status = None
            complete = False
            error_type = None
            try:
                request = urllib.request.Request(base_url.rstrip('/') + '/rollout/task/' + urllib.parse.quote(task_id, safe=''), method='GET')
                reads += 1
                with raw_path.open('xb') as output:
                    try:
                        response = opener(request, timeout=max(.001, min(10, deadline - time.monotonic())))
                    except urllib.error.HTTPError as error:
                        response = error
                    with response:
                        status = response.status
                        while chunk := response.read(65536):
                            output.write(chunk)
                            digest.update(chunk)
                body = json.loads(raw_path.read_bytes())
                if status == 200 and body.get('task_id') == task_id:
                    complete = True
                    responses[task_id] = {'path': str(raw_path), 'sha256': digest.hexdigest(),
                                          'terminal_task': body.get('status') in {'completed', 'failed'}}
                elif status == 200:
                    raise ValueError('Coordinator response belongs to a different task')
            except Exception as error:
                error_type = type(error).__name__
                errors.append({'task_id': task_id, 'error_type': error_type})
            finally:
                if raw_path.exists():
                    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                else:
                    raw_sha = None
                receipt_path.write_text(json.dumps({'task_id': task_id, 'started': started, 'finished': time.time(),
                    'HTTP_status': status, 'raw_sha256': raw_sha, 'complete_task_JSON': complete,
                    'error_type': error_type, 'operation': 'read_only_coordinator_GET'}, indent=2))
            if time.monotonic() >= deadline:
                break
    except TimeoutError:
        errors.append({'error_type': 'TimeoutError', 'scope': 'bounded metadata export'})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    index_raw = json.dumps(index, ensure_ascii=False, indent=2).encode()
    history = directory / 'indices'
    history.mkdir(exist_ok=True)
    immutable = history / (hashlib.sha256(index_raw).hexdigest() + '.json')
    if not immutable.exists():
        immutable.write_bytes(index_raw)
    temporary = latest.with_suffix('.tmp')
    temporary.write_bytes(index_raw)
    temporary.replace(latest)
    return {'questions': len(manifest['cases']), 'captured_task_responses': len(responses),
            'terminal_task_responses': sum(row['terminal_task'] for row in responses.values()),
            'coordinator_GETs_this_batch': reads, 'errors': errors, 'new_model_or_judge_calls': 0,
            'index': str(latest), 'immutable_index': str(immutable)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:19170')
    parser.add_argument('--seconds', type=float, default=35)
    args = parser.parse_args()
    print(json.dumps(collect(args.manifest.read_bytes(), args.directory, args.url, args.seconds), indent=2))


if __name__ == '__main__':
    main()
