"""Replace the coordinator while the sampler is paused before a new task."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from handoff_training_v6 import get,post,fully_drained

ROOT=Path('/mount-data/CLM-Replication')
DIRECTORY=ROOT/'management/queue-timeout-recovery-v2'
GATE=DIRECTORY/'PAUSE_NEW_TASKS.json'
PAUSED=ROOT/'runs/c2-seed1234/audit/control-admission-00000-000-010-paused.json'

def identity(pid):
    p=Path('/proc')/str(pid)
    return {'pid':pid,'start_ticks':int((p/'stat').read_text().split()[21]),
            'argv':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()}

def main():
    assert json.loads((DIRECTORY/'TERMINAL_RECOVERY_CPU_ACCEPTED.json').read_bytes())['accepted']
    assert json.loads((DIRECTORY/'QUEUE_CPU_ACCEPTED.json').read_bytes())['accepted']
    pins=json.loads((DIRECTORY/'SOURCE_PINS.json').read_bytes())
    for name,expected in pins['files'].items():assert hashlib.sha256((DIRECTORY/name).read_bytes()).hexdigest()==expected
    for name,expected in pins['external_files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected
    with (DIRECTORY/'TERMINAL_HANDOFF_INTENT.json').open('x') as handle:
        json.dump({'created':time.time(),'sampler_source':'training-v7','training_or_model_restart':False,
                   'all_original_tasks_must_finish':True},handle,indent=2)
    deadline=time.monotonic()+7200
    while not PAUSED.exists():
        current=json.loads((ROOT/'runs/c2-seed1234/CURRENT_OWNER.json').read_bytes())
        assert current['attempt']==7 and (Path('/proc')/str(current['pid'])).exists()
        if (Path(current['directory'])/'EXIT.json').exists():raise RuntimeError('Sampler exited before the admission boundary')
        if time.monotonic()>deadline:raise TimeoutError('Original task is still in its execution budget')
        time.sleep(2)
    assert GATE.exists()
    assert not (ROOT/'runs/c2-seed1234/audit/rollout-00000/candidate-000/group-010-intent.json').exists()
    assert not (ROOT/'runs/c2-seed1234/audit/sample-summary-policy-00000.json').exists()
    while True:
        state=get('http://127.0.0.1:19170/rollout/status')
        health=get('http://127.0.0.1:19171/health');sessions=get('http://127.0.0.1:19171/sessions')['sessions']
        inference=get('http://127.0.0.1:19171/admin/inference/status')
        if state['pipeline']['pending_sessions']==0 and all(x=='completed' for x in state['tasks'].values()) and fully_drained(health,sessions) and inference['inflight']==0:break
        if time.monotonic()>deadline:raise TimeoutError('Original callbacks or archives are not yet closed')
        time.sleep(.5)
    before_receipts=len(list((ROOT/'runs/token-ledger').glob('*.json')))
    export=subprocess.run([str(ROOT/'management/runtime.sh'),'env','CUDA_VISIBLE_DEVICES=', 'python',str(DIRECTORY/'export_snapshot.py')],capture_output=True,text=True)
    (DIRECTORY/'EXPORT_RECEIPT.json').write_text(json.dumps({'exit_code':export.returncode,'stdout':export.stdout,'stderr':export.stderr},indent=2))
    assert export.returncode==0,export.stderr
    previous=identity(2345487);assert 'serve_rollout' in previous['argv'] and 'CLM-Replication' in previous['argv']
    os.kill(2345487,signal.SIGTERM)
    stop=time.monotonic()+45
    while Path('/proc/2345487').exists():
        if time.monotonic()>stop:raise TimeoutError('Old coordinator has not closed')
        time.sleep(.25)
    (DIRECTORY/'PREVIOUS_COORDINATOR.json').write_text(json.dumps({'identity':previous,'signal':15,'exactgone':True,
        'native_wait_exit_code':None,'reason':'original coordinator had no owned wait wrapper'},indent=2))
    with (DIRECTORY/'COORDINATOR_DISPATCH_INTENT.json').open('x') as handle:
        json.dump({'created':time.time(),'source_pins_sha256':hashlib.sha256((DIRECTORY/'SOURCE_PINS.json').read_bytes()).hexdigest()},handle,indent=2)
    with (DIRECTORY/'owner.log').open('wb') as log:
        process=subprocess.Popen([str(ROOT/'env/rl/bin/python'),str(DIRECTORY/'owned_coordinator.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (DIRECTORY/'COORDINATOR_LAUNCH.json').write_text(json.dumps({'pid':process.pid,'created':time.time()},indent=2))
    start=time.monotonic()+240
    while True:
        try:
            assert get('http://127.0.0.1:19170/health')['status']=='ok';break
        except (OSError,AssertionError):
            if process.poll() is not None:raise RuntimeError('New coordinator exited before readiness')
            if time.monotonic()>start:raise TimeoutError('Coordinator readiness')
            time.sleep(.5)
    plan=json.loads((DIRECTORY/'snapshot/PLAN.json').read_bytes());assert plan['original_pending_sessions']==0
    for item in plan['tasks']:
        task_id=item['metadata']['task_id']
        assert get('http://127.0.0.1:19170/rollout/task/'+task_id)==json.loads((DIRECTORY/'snapshot'/item['status_file']).read_bytes())
        assert get('http://127.0.0.1:19170/tasks/'+task_id+'/sessions')['sessions']==item['sessions']
    topology=yaml.safe_load((ROOT/'management/topology-training-v7.yaml').read_text())
    node=topology['gateway']['nodes'][0]
    registration={'node_id':node['id'],'gateway_url':node['public_url'],'max_init_workers':node['max_init_workers'],
                  'max_run_workers':node['max_run_workers'],'max_postrun_workers':node['max_postrun_workers'],
                  'heartbeat_interval_seconds':topology['gateway']['heartbeat_interval_seconds']}
    post('http://127.0.0.1:19170/nodes/register',registration)
    current=json.loads((ROOT/'runs/c2-seed1234/CURRENT_OWNER.json').read_bytes())
    assert current['attempt']==7 and identity(current['pid'])['argv'].find('owned_training_v7.py')>=0
    assert get('http://10.10.15.17:15000/model_info')['weight_version']=='1'
    assert len(list((ROOT/'runs/token-ledger').glob('*.json')))==before_receipts
    proof={'accepted':True,'created':time.time(),'terminal_task_statuses_exact':len(plan['tasks']),
           'terminal_session_summaries_exact':True,'original_timeout_results_retained':91,
           'old_coordinator':previous,'new_coordinator_owner':identity(process.pid),
           'training_owner':identity(current['pid']),'training_and_model_restarted':False,
           'model_calls_during_switch':0,'optimizer_updates':0,'next_unstarted_position':10}
    (DIRECTORY/'TERMINAL_ROOT_ACCEPTED.json').write_text(json.dumps(proof,indent=2))
    GATE.unlink()
    resumed=ROOT/'runs/c2-seed1234/audit/control-admission-00000-000-010-resumed.json'
    limit=time.monotonic()+10
    while not resumed.exists():
        if time.monotonic()>limit:raise TimeoutError('Sampler did not leave the verified admission boundary')
        time.sleep(.2)
    (DIRECTORY/'HANDOFF_COMPLETE.json').write_text(json.dumps({'created':time.time(),'accepted':True,
        'new_task_admission_resumed':True,'scientific_source':'training-v7','model_tool_replay':0},indent=2))
    print(json.dumps(proof,indent=2),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        (DIRECTORY/'TERMINAL_HANDOFF_ERROR.json').write_text(json.dumps({'created':time.time(),'type':type(error).__name__,'message':str(error)},indent=2));raise
