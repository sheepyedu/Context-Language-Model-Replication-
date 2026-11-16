"""Read a compact current-owner snapshot without model requests or mutations."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

root = Path('/mount-data/CLM-Replication')
run = root / 'runs/c2-seed1234'
owner = json.loads((run / 'CURRENT_OWNER.json').read_bytes())
directory = Path(owner['directory'])

def identity(pid):
    proc = Path('/proc') / str(pid)
    try:
        stat = (proc / 'stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        raw = (proc / 'cmdline').read_bytes()
        return {'pid': pid, 'uid': proc.stat().st_uid, 'start_ticks':int(fields[19]),
                'state':fields[0], 'cmdline':raw.replace(b'\0',b' ').decode(errors='replace'),
                'raw_sha256':hashlib.sha256(raw).hexdigest()}
    except FileNotFoundError:
        return {'pid':pid,'gone':True}

with (directory / 'controller.log').open('rb') as handle:
    handle.seek(0,2)
    size = handle.tell()
    handle.seek(max(0,size-131072))
    tail = handle.read().decode(errors='replace')
phases=[]
for label in ['ref forward','actor forward','actor train','train actor']:
    values=re.findall(re.escape(label)+r':[^\r\n]*',tail)
    if values: phases.append(values[-1])
metrics=[line for line in tail.splitlines() if 'train/grad_norm' in line and 'train/step' in line]
proofs=[]
for p in sorted(directory.glob('backward-preflight-rank*/ACCEPTED.json')):
    value=json.loads(p.read_text())
    proofs.append({'rank':value['rank'],'accepted':value['accepted'],'optimizer_steps':value['optimizer_steps'],
                   'peak_allocated_bytes':value['peak_allocated_bytes'],'peak_reserved_bytes':value['peak_reserved_bytes']})
exit_path=directory/'EXIT.json'
print(json.dumps({'observed':time.time(),'owner':owner,'actual_owner':identity(owner['pid']),
                  'exit':json.loads(exit_path.read_text()) if exit_path.exists() else None,
                  'preflight':proofs,'native_progress':phases,'native_optimizer_metrics':metrics[-3:],
                  'log_bytes':size,'log_tail':tail[-1800:],
                  'gpus':subprocess.check_output(['nvidia-smi','-i','4,5,6,7','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader'],text=True)},indent=2))
