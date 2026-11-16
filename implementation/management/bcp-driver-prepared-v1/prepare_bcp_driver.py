"""Prepare final-test driver metadata and matching retrieval instructions."""
import hashlib
import json
from pathlib import Path
import shutil
import time

import yaml

ROOT = Path('/mount-data/CLM-Replication')
PREPARED = ROOT / 'management/bcp-driver-prepared-v1'
BASE = ROOT / 'management/retrieval-gateway-prepared-v2/source/clm_replication'
TARGET = PREPARED / 'source/clm_replication'
assert not TARGET.exists(), 'Inspect the existing preparation before retrying'
shutil.copytree(BASE, TARGET, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
original = (BASE / 'polar_agent.py').read_text()
lines = original.splitlines(keepends=True)
notice_lines = [line for line in lines if 'Public web retrieval is available' in line]
assert len(notice_lines) == 1
public_notice = (
    '\nPublic web retrieval is available through Bash. Use: '
    '/mount-data/CLM-Replication/env/agent/bin/python -m clm_replication.web search '
    '"your query" or fetch "https://example.org/page". Search returns up to ten '
    'results, fetch returns up to 32768 characters of page text.'
)
fixed_notice = (
    '\nRetrieval searches the fixed BrowseComp-Plus document corpus through Bash. Use: '
    '/mount-data/CLM-Replication/env/agent/bin/python -m clm_replication.web search '
    '"your query". Search returns up to ten results with snippets of at most 512 '
    'Qwen tokens. Read a returned document with the same command using fetch '
    '"bcp://DOCUMENT_ID". Reads return up to 8192 Qwen tokens of source text.'
)
selection = (
    "        profile = self.settings.get('retrieval_profile', 'public_web')\n"
    f"        notices = {{'public_web': {public_notice!r},\n"
    f"                   'browsecomp_plus_fixed': {fixed_notice!r}}}\n"
    "        if profile not in notices:\n"
    "            raise ValueError('Unknown driver retrieval profile')\n"
    "        retrieval_notice = notices[profile]\n"
)
updated = original.replace('        encoded = json.dumps(instruction, ensure_ascii=False)\n',
                           selection + '        encoded = json.dumps(instruction, ensure_ascii=False)\n')
updated = updated.replace(notice_lines[0], '            f"text += {retrieval_notice!r}\\n"\n')
(TARGET / 'polar_agent.py').write_text(updated)

config = yaml.safe_load((ROOT / 'management/train-config-v7.yaml').read_text())
config['clm_run_id'] = 'c2-seed1234-browsecomp-plus-final'
config['clm_audit_dir'] = str(ROOT / 'runs/c2-seed1234/browsecomp-plus-final/audit')
config['clm_phase'] = 'final_test'
config['polar_eval_dataset_name'] = 'browsecomp_plus'
config['polar_task_id_template'] = 'clm-bcp-final-{rollout_id}-{sample.metadata.qid}'
task = config['polar_task_template']
task['metadata'].update(dataset='browsecomp_plus', retrieval_profile='browsecomp_plus_fixed')
task['agent']['settings'] = {'retrieval_profile': 'browsecomp_plus_fixed'}
volumes = task['runtime']['kwargs']['volumes']
old = str(ROOT / 'agent-bundle-v1/clm_replication') + ':'
assert sum(volume.startswith(old) for volume in volumes) == 1
task['runtime']['kwargs']['volumes'] = [
    volume.replace(old, str(TARGET) + ':', 1) if volume.startswith(old) else volume
    for volume in volumes
]
(PREPARED / 'browsecomp-plus-final.yaml').write_text(yaml.safe_dump(config, sort_keys=False))

data = ROOT / 'data/browsecomp-plus-chat.jsonl'
manifest = json.loads((ROOT / 'data/BROWSECOMP_TEST_MANIFEST.json').read_text())
assert hashlib.sha256(data.read_bytes()).hexdigest() == manifest['sha256']
rows = [json.loads(line) for line in data.read_text().splitlines()]
assert len(rows) == 830 and len({row['metadata']['qid'] for row in rows}) == 830
assert all(set(row) == {'prompt', 'label', 'metadata'} for row in rows)
proof = {
    'prepared_only': True, 'created': time.time(), 'production_activation': False,
    'dataset_path': str(data), 'dataset_sha256': manifest['sha256'], 'count': 830,
    'original_rows_changed': False, 'gold_evidence_documents_added': False,
    'model_calls': 0, 'base_harness_sha256': hashlib.sha256(original.encode()).hexdigest(),
    'candidate_harness_sha256': hashlib.sha256(updated.encode()).hexdigest(),
    'gateway_matches_accepted_candidate': (BASE / 'gateway.py').read_bytes() == (TARGET / 'gateway.py').read_bytes(),
}
(PREPARED / 'PREPARED.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof, indent=2))
