"""A finite admission lease compatible with the running path-exists protocol.

The gate is a symlink through a live process directory descriptor. Expiry or
process death makes Path.exists() false, including SIGKILL. No job polling or
model requests are performed by the lease owner.
"""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import time
import uuid


class LeaseEnded(Exception):
    def __init__(self, signum):
        self.signum = signum


@contextmanager
def gate_lock(gate):
    with gate.with_name(gate.name + '.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_lease(gate, directory, duration_seconds):
    if not 0 < duration_seconds <= 3600:
        raise ValueError('Admission lease duration must be within one hour')
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    body_directory = directory / 'body'
    body_directory.mkdir(mode=0o700)
    gate.parent.mkdir(parents=True, exist_ok=True)
    lease_id = uuid.uuid4().hex
    created = time.time()
    identity = {'pid': os.getpid(), 'uid': os.getuid(),
                'start_ticks': int(Path('/proc/self/stat').read_text().split()[21])}
    payload = {'lease_id': lease_id, 'created': created,
               'expires_at_unix': created + duration_seconds,
               'maximum_engineering_seconds': duration_seconds,
               'owner': identity, 'record_directory': str(directory)}
    (directory / 'INTENT.json').write_text(json.dumps(payload, indent=2))
    descriptor = None
    target = None
    published = False
    reason = 'error'
    exit_code = 1

    def end(signum, frame):
        raise LeaseEnded(signum)

    for signum in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, end)
    signal.setitimer(signal.ITIMER_REAL, duration_seconds)
    try:
        body = body_directory / (lease_id + '.json')
        body.write_text(json.dumps(payload, indent=2))
        body.chmod(0o600)
        descriptor = os.open(body_directory, os.O_RDONLY | os.O_DIRECTORY)
        target = f'/proc/{os.getpid()}/fd/{descriptor}/{body.name}'
        with gate_lock(gate):
            if gate.is_symlink() and not gate.exists():
                gate.unlink()
            os.symlink(target, gate)
        published = True
        (directory / 'PUBLISHED.json').write_text(json.dumps({
            **payload, 'gate': str(gate), 'symlink_target': target,
        }, indent=2))
        while True:
            signal.pause()
    except LeaseEnded as ended:
        reason = 'expired' if ended.signum == signal.SIGALRM else 'released'
        exit_code = 0
    except Exception as error:
        (directory / 'ERROR.json').write_text(json.dumps({
            'error_type': type(error).__name__, 'message': str(error),
        }, indent=2))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        # Close the descriptor first. The native sampler resumes even if disk
        # cleanup is delayed or the owner is killed before writing EXIT.json.
        if descriptor is not None:
            os.close(descriptor)
        if published:
            with gate_lock(gate):
                if gate.is_symlink() and os.readlink(gate) == target:
                    gate.unlink()
        (directory / 'EXIT.json').write_text(json.dumps({
            'created': time.time(), 'exit_code': exit_code, 'reason': reason,
            'published': published, 'lease_id': lease_id,
            'new_model_calls': 0, 'job_polling': False,
        }, indent=2))
    return exit_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--duration-seconds', type=int, required=True)
    args = parser.parse_args()
    raise SystemExit(run_lease(args.gate, args.directory, args.duration_seconds))


if __name__ == '__main__':
    main()
