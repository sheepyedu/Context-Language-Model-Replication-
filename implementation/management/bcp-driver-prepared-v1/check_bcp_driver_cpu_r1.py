"""Check actual native task rendering and harness instruction construction."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import yaml

ROOT = Path('/mount-data/CLM-Replication')
PREPARED = ROOT / 'management/bcp-driver-prepared-v1'
sys.path[:0] = [str(PREPARED / 'source'), str(ROOT / 'src/ProRL/src')]
from polar.agent.models import AgentSpec
from slime_bridge.config import resolve_polar_slime_config
from slime_bridge.rollout import _build_task_payload
from clm_replication.polar_agent import ClmHarness
from clm_replication.retrieval_routing import retrieval_backend

spec = importlib.util.spec_from_file_location('original_harness', ROOT / 'source-revisions/training-v10/clm_replication/polar_agent.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
config = yaml.safe_load((PREPARED / 'browsecomp-plus-final.yaml').read_text())
args = SimpleNamespace(**config, rollout_batch_size=1, n_samples_per_prompt=1)
native = resolve_polar_slime_config(args)
question = 'Generic fixture with quotes " and shell $(false), Unicode 中文.'
label = 'PRIVATE_LABEL_FIXTURE_DO_NOT_EXPOSE'
sample = SimpleNamespace(prompt=[{'role': 'user', 'content': question}], label=label,
                         metadata={'qid': 'fixture-id'}, index=0, group_index=0)
payload = _build_task_payload(args=args, config=native, group=[sample], rollout_id=0, task_position=0)
checks = []
assert payload['instruction'] == question
checks.append('native_payload_preserves_original_question')
assert payload['metadata']['dataset'] == 'browsecomp_plus'
assert payload['metadata']['retrieval_profile'] == 'browsecomp_plus_fixed'
checks.append('native_driver_metadata_selects_fixed_corpus')
assert retrieval_backend(payload['metadata']).endswith(':19181/retrieve')
checks.append('native_payload_routes_to_fixed_backend')
assert payload['evaluator']['config']['expected_answer'] == label
assert label not in payload['instruction'] and label not in json.dumps(payload['agent'])
checks.append('label_only_in_evaluator_configuration')
assert payload['num_samples'] == 1 and payload['task_id'] == 'clm-bcp-final-0-fixture-id'
checks.append('native_single_session_and_task_identity')

agent = AgentSpec.model_validate(payload['agent'])
harness = ClmHarness(agent)
command = harness.run_steps(payload['instruction'])[0].command
assert 'bcp://DOCUMENT_ID' in command and '8192 Qwen tokens' in command
assert 'https://example.org/page' not in command and 'Public web retrieval' not in command
checks.append('fixed_harness_describes_actual_tool_contract')
script = command.split("<<'CLM_INSTRUCTION_JSON'\n", 1)[1].split('\nCLM_INSTRUCTION_JSON', 1)[0]
with tempfile.TemporaryDirectory() as temporary:
    target = Path(temporary) / 'instruction.txt'
    fixture_script = script.replace('/polar/session/instruction.txt', str(target))
    completed = subprocess.run([sys.executable, '-c', fixture_script], capture_output=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
    instruction = target.read_text()
assert instruction.startswith(question + '\n\n') and label not in instruction
checks.append('actual_heredoc_preserves_question_without_shell_expansion')
assert '/polar/session/artifacts/answer.txt' in instruction
checks.append('original_answer_artifact_contract_preserved')

for settings in [{}, {'retrieval_profile': 'public_web'}]:
    public_agent = AgentSpec(import_path='clm_replication.polar_agent:ClmHarness',
                             model_name='qwen35-clm', settings=settings)
    assert ClmHarness(public_agent).run_steps(question)[0].command == original.ClmHarness(public_agent).run_steps(question)[0].command
checks.append('default_and_explicit_public_harness_commands_byte_equal')
try:
    bad = AgentSpec(import_path='clm_replication.polar_agent:ClmHarness', settings={'retrieval_profile': 'unregistered'})
    ClmHarness(bad).run_steps(question)
except ValueError:
    pass
else:
    raise AssertionError('Unknown profile accepted')
checks.append('unknown_harness_profile_rejected')
assert config['clm_phase'] == 'final_test' and config['polar_eval_dataset_name'] == 'browsecomp_plus'
checks.append('final_test_identity_separate_from_training')
volumes = payload['runtime']['kwargs']['volumes']
assert any(volume.startswith(str(PREPARED / 'source/clm_replication') + ':') and volume.endswith(':ro') for volume in volumes)
checks.append('candidate_agent_bundle_read_only_mount')
baseline = yaml.safe_load((ROOT / 'management/train-config-v7.yaml').read_text())
assert payload['runtime']['image'] == baseline['polar_task_template']['runtime']['image']
assert payload['timeout_seconds'] == baseline['polar_task_template']['timeout_seconds']
checks.append('original_runtime_image_and_task_budget_preserved')
assert native.task_template == config['polar_task_template']
checks.append('rendering_does_not_mutate_template')
proof = {'accepted': True, 'created': time.time(), 'count': len(checks), 'checks': checks,
         'native_cli_fixture_integer_exit': completed.returncode, 'model_or_judge_calls': 0,
         'production_activation': False, 'actual_model_consumer_pending': True,
         'payload_fixture_only': True, 'dataset_test_answers_used_in_fixture': False}
(PREPARED / 'CPU_ACCEPTED.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof, indent=2))
