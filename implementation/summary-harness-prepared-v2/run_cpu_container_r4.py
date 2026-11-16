"""Run only mock-completion/Bash fixtures in the existing native task image."""
import json
from pathlib import Path
import subprocess
import time

root=Path('/mount-data/CLM-Replication')
stage=root/'management/summary-harness-prepared-v2/cpu-container-r4'
stage.mkdir(exist_ok=False)
workspace=stage/'workspace';workspace.mkdir();workspace.chmod(0o777)
name='clm-summary-cpu-v2-20261116-r4'
argv=[str(root/'management/runtime.sh'),'docker','run','--name',name,'--rm','--network','none',
      '--label','clm.fixture=summary-cpu-v2','--cpus','2','--memory','4g','--pids-limit','256',
      '--user','1005:1005','--workdir','/work','--read-only',
      '--tmpfs','/tmp:rw,nosuid,nodev,size=256m,uid=1005,gid=1005',
      '-e','PYTHONDONTWRITEBYTECODE=1','-e','HF_HUB_OFFLINE=1',
      '-e','LITELLM_LOCAL_MODEL_COST_MAP=true','-e','OPENAI_API_KEY=dummy',
      '-e','PYTHONPATH=/mount-data/CLM-Replication/env/summary-agent-deps:/mount-data/CLM-Replication/src:/mount-data/CLM-Replication/src/context-language-models/clm',
      '-v',f'{workspace}:/work:rw']
mounts={root/'env/agent':root/'env/agent',root/'env/cpython312':root/'env/cpython312',
        root/'env/summary-agent-deps':root/'env/summary-agent-deps',
        root/'summary-assets/tokenizer':root/'summary-assets/tokenizer',
        root/'agent-bundle-summary-v4/clm_replication':root/'src/clm_replication',
        root/'src/context-language-models/clm/clm_harness':root/'src/context-language-models/clm/clm_harness',
        root/'management/summary-harness-prepared-v2':Path('/prepared')}
for source,target in mounts.items():argv.extend(['-v',f'{source}:{target}:ro'])
argv.extend(['--entrypoint',str(root/'env/agent/bin/python'),
             'sha256:a7e768658111a21f942395b89214267138b74dc6a8f37f91f37772239812e19c',
             '/prepared/check_summary_loop.py'])
(stage/'INTENT.json').write_text(json.dumps({'created':time.time(),'argv':argv,'model_calls':0,'kind':'CPU fixtures only'},indent=2))
with (stage/'OUTPUT.log').open('wb') as log:
    child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
    (stage/'LAUNCH.json').write_text(json.dumps({'pid':child.pid,'created':time.time()}))
    code=child.wait(timeout=180)
(stage/'EXIT.json').write_text(json.dumps({'exit_code':code,'finished':time.time(),'pid':child.pid}))
print(json.dumps({'exit_code':code,'container':name,'proof':str(workspace/'SUMMARY_LOOP_CPU_ACCEPTED.json')}))
raise SystemExit(code)
