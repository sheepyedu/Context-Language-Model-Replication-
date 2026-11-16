"""Read native progress and bounded full-log optimizer evidence."""

from collections import deque
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


ROOT = Path('/mount-data/CLM-Replication')


def scan_optimizer_metrics(path, max_seconds=30, clock=time.monotonic):
    deadline = clock() + max_seconds
    snapshot_bytes = path.stat().st_size
    rows = deque(maxlen=3)
    matching_lines = 0
    scanned_bytes = 0
    with path.open('rb') as handle:
        while scanned_bytes < snapshot_bytes:
            if clock() >= deadline:
                break
            raw = handle.readline(snapshot_bytes - scanned_bytes)
            if not raw:
                break
            scanned_bytes += len(raw)
            if b'train/grad_norm' in raw and b'train/step' in raw:
                rows.append(raw.decode(errors='replace').rstrip('\r\n'))
                matching_lines += 1
    return list(rows), {
        'complete': scanned_bytes == snapshot_bytes,
        'scanned_bytes': scanned_bytes,
        'snapshot_bytes': snapshot_bytes,
        'matching_lines': matching_lines,
        'matching_lines_are_not_an_accepted_optimizer_update_count': True,
    }


def identity(pid):
    proc = Path('/proc') / str(pid)
    try:
        stat = (proc / 'stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        raw = (proc / 'cmdline').read_bytes()
        return {'pid': pid, 'uid': proc.stat().st_uid, 'start_ticks': int(fields[19]),
                'state': fields[0], 'cmdline': raw.replace(b'\0', b' ').decode(errors='replace'),
                'raw_sha256': hashlib.sha256(raw).hexdigest()}
    except FileNotFoundError:
        return {'pid': pid, 'gone': True}


def main():
    run = ROOT / 'runs/c2-seed1234'
    owner = json.loads((run / 'CURRENT_OWNER.json').read_bytes())
    directory = Path(owner['directory'])
    log = directory / 'controller.log'
    with log.open('rb') as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 131072))
        tail = handle.read().decode(errors='replace')
    phases = []
    for label in ['ref forward', 'actor forward', 'actor train', 'train actor']:
        values = re.findall(re.escape(label) + r':[^\r\n]*', tail)
        if values:
            phases.append(values[-1])
    metrics, scan = scan_optimizer_metrics(log)
    proofs = []
    for path in sorted(directory.glob('backward-preflight-rank*/ACCEPTED.json')):
        value = json.loads(path.read_text())
        proofs.append({'rank': value['rank'], 'accepted': value['accepted'],
                       'optimizer_steps': value['optimizer_steps'],
                       'peak_allocated_bytes': value['peak_allocated_bytes'],
                       'peak_reserved_bytes': value['peak_reserved_bytes']})
    exit_path = directory / 'EXIT.json'
    print(json.dumps({
        'observed': time.time(), 'owner': owner, 'actual_owner': identity(owner['pid']),
        'exit': json.loads(exit_path.read_text()) if exit_path.exists() else None,
        'preflight': proofs, 'native_progress': phases, 'native_optimizer_metrics': metrics,
        'optimizer_metrics_scan': scan, 'log_bytes': size, 'log_tail': tail[-1800:],
        'gpus': subprocess.check_output([
            'nvidia-smi', '-i', '4,5,6,7',
            '--query-gpu=index,uuid,utilization.gpu,memory.used',
            '--format=csv,noheader'], text=True),
    }, indent=2))


if __name__ == '__main__':
    main()
