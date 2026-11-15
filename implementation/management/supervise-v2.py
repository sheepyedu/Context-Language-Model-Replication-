"""One-shot observation of this run, without launching or restarting work."""
import ast
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.request

from clm_replication.accounting import summarize

ROOT=Path('/mount-data/CLM-Replication')
RUN=ROOT/'runs/c2-seed1234'

def get_json(url):
    try:
        with urllib.request.urlopen(url,timeout=15) as response:
            return json.load(response)
    except Exception as error:
        return {'observation_error':type(error).__name__,'message':str(error)}

def identity(pid):
    path=Path('/proc')/str(pid)
    if not path.exists():return {'pid':pid,'present':False}
    try:
        return {'pid':pid,'present':True,'start_ticks':int((path/'stat').read_text().split()[21]),
                'argv':(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(),
                'uid':int((path/'status').read_text().split('Uid:')[1].split()[0])}
    except FileNotFoundError:return {'pid':pid,'present':False}

launch=json.loads((RUN/'CURRENT_OWNER.json').read_text())
directory=Path(launch['directory'])
log_path=directory/'controller.log'
updates=[]
for line in log_path.open(errors='replace'):
    match=re.search(r'(?<!critic-)step (\d+): (\{.*\})',line)
    if match and 'train/grad_norm' in line and 'train/step' in line:
        raw_metrics = match.group(2).split('\x1b')[0]
        try:
            value = ast.literal_eval(raw_metrics)
        except (ValueError, SyntaxError):
            value = {'step': int(match.group(1)), 'unparsed_metrics': raw_metrics,
                     'requires_inspection': True}
        updates.append(value)
intents=sorted((RUN/'audit').glob('rollout-*/candidate-*/group-*-intent.json'),key=lambda p:p.stat().st_mtime)
closed=list((RUN/'audit').glob('rollout-*/candidate-*/group-*-result.json'))
current=None
for intent in intents:
    result=intent.with_name(intent.name.replace('-intent.json','-result.json'))
    if result.exists():continue
    payload=json.loads(intent.read_text())
    status=get_json('http://127.0.0.1:19170/rollout/task/'+payload['task_id'])
    grades=Counter()
    for session in status.get('results',[]):
        trajectory=session.get('trajectory') or {}
        outcome=trajectory.get('metadata',{}).get('evaluation',{})
        grades[str(outcome.get('outcome_reward'))]+=1
    current={'task_id':payload['task_id'],'status':status.get('status'),
             'completed_sessions':status.get('completed_sessions'),'total_sessions':status.get('total_sessions'),
             'reward_counts':dict(grades),'error':status.get('observation_error')}
    break
native=get_json('http://127.0.0.1:19170/rollout/status')
native_tasks={key:value for key,value in native.get('tasks',{}).items() if key.startswith('clm-'+RUN.name+'-')}
sessions=get_json('http://127.0.0.1:19171/sessions')
summary=summarize(ROOT)
applications=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
owner=identity(launch['pid'])
exit_path=directory/'EXIT.json'
report={'created':time.time(),'attempt':launch['attempt'],'owner':owner,
        'exit':json.loads(exit_path.read_text()) if exit_path.exists() else None,
        'native_task_status_counts':dict(Counter(native_tasks.values())),
        'native_tasks':native_tasks,
        'pipeline_pending_sessions':native.get('pipeline',{}).get('pending_sessions'),
        'router':identity(json.loads((directory/'ROUTER_LAUNCH.json').read_text())['pid']),
        'trainer_child':identity(json.loads((directory/'CHILD_LAUNCH.json').read_text())['pid']),
        'current_group':current,'closed_prompt_groups':len(closed),'intent_prompt_groups':len(intents),
        'submitted_prompt_groups':sum(p.with_name(p.name.replace('-intent.json','-submit.json')).exists() for p in intents),
        'optimizer_metric_records':updates,'recorded_optimizer_steps':len(updates),
        'optimizer_count_scope':'Current-attempt native actor train/grad_norm and train/step rows, pending parameter/sync/restore verification.',
        'serving_model':get_json('http://10.10.15.17:15000/model_info'),
        'gateway':get_json('http://127.0.0.1:19171/admin/inference/status'),
        'session_status_counts':dict(Counter(s['status'] for s in sessions.get('sessions',[]))),
        'tokens':{'all':summary['all'],'by_role':summary['by_role']},
        'gpu45':[line for line in gpus.splitlines() if line.split(',')[0].strip() in ('4','5')],
        'gpu45_processes':[line for line in applications.splitlines() if any(x in line for x in (
            'GPU-101b4bf3-3f4e-5706-eef9-7e3d38ba4ea3','GPU-c64fa322-ea6c-ed73-b81b-67bddfd2ad9e'))],
        'free_bytes':shutil.disk_usage(ROOT).free}
output=ROOT/'management/supervision'
output.mkdir(exist_ok=True)
name=output/f'{time.time_ns()}.json'
name.write_text(json.dumps(report,indent=2))
print(json.dumps({**report,'evidence':str(name)},indent=2))
