"""Regression against a real closed timeout, plus an unresolved-call fixture."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from polar.rollout.models import TaskResult
from slime_bridge.adapter import session_result_to_samples
from clm_replication.native_failures import training_view, restore_native_metadata

root = Path('/mount-data/CLM-Replication')
path = root/'runs/c2-seed1234/audit/rollout-00000/candidate-000/group-000-result.json'
result = TaskResult.model_validate_json(path.read_text())
sid = 'sk-polar-91916f2c-143d-44af-8d17-d3be4a5bd24d'
source = next(s for s in result.results if s.session_id == sid)
original = source.model_dump_json()
red = session_result_to_samples(source, 0, trajectory_index=0, reward_key='score', max_tokens=32768)
assert len(red) == 6 and sum(sum(s.loss_mask) for s in red) == 0
single = result.model_copy(update={'results': [source]})
view, disposition = training_view(single)
green = session_result_to_samples(view.results[0], 0, trajectory_index=0, reward_key='score', max_tokens=32768)
restore_native_metadata(green, disposition)
assert len(green) == 6 and sum(sum(s.loss_mask) for s in green) == sum(len(t.response_ids) for t in source.trajectory.traces)
assert all(s.metadata['polar']['session_status'] == 'ERROR' for s in green)
assert all(s.status.name == 'COMPLETED' for s in green)
assert source.model_dump_json() == original
reward = source.trajectory.metadata['evaluation']['outcome_reward']
assert reward == -0.2 and all(s.reward['score'] == reward for s in green)
with tempfile.TemporaryDirectory(prefix='clm-closed-failure-test-') as temporary:
    fixture = Path(temporary)
    archive = fixture/'runs/session-artifacts'/sid
    shutil.copytree(root/'runs/session-artifacts'/sid/'logs/agent', archive/'logs/agent')
    requests = fixture/'runs/provider-attempts'/sid
    for index, p in enumerate(sorted((root/'runs/provider-attempts'/sid).glob('*/INTENT.json'))):
        directory = requests/str(index)
        directory.mkdir(parents=True)
        shutil.copyfile(p, directory/'INTENT.json')
        if index:
            shutil.copyfile(p.parent/'RESPONSE.json', directory/'RESPONSE.json')
    unknown_view, unknown_disposition = training_view(single, root=fixture)
    assert not unknown_disposition and unknown_view.results[0].status == 'ERROR'
judge_error = next(s for s in result.results if s.error and 'evaluator failed:' in s.error)
invalid = result.model_copy(update={'results': [judge_error]})
_, invalid_disposition = training_view(invalid)
assert not invalid_disposition
proof = {'passed': 4, 'source_session': sid, 'model_calls': len(green),
         'red_trainable_tokens': 0, 'green_trainable_tokens': sum(sum(s.loss_mask) for s in green),
         'original_negative_reward': reward, 'original_native_record_unchanged_sha256': hashlib.sha256(original.encode()).hexdigest(),
         'unresolved_provider_attempt_stays_invalid': True, 'invalid_judge_stays_invalid': True,
         'new_model_tool_or_judge_calls': 0}
(root/'management/training-v5-recovery/NATIVE_FAILURE_CPU.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof,indent=2))
