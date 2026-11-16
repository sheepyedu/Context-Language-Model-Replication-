"""Native host-side Summary wiring fixtures, without requests or model loading."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path('/mount-data/CLM-Replication')
PREPARED=ROOT/'management/summary-harness-prepared-v2'
sys.path[:0]=[str(PREPARED/'summary-host-source-v4'),str(ROOT/'source-revisions/training-v10'),str(ROOT/'src')]
import yaml
from polar.agent.models import AgentSpec, AgentRunResult
from polar.trajectory.models import CompletionRecord, CompletionSession
from polar.trajectory.builder.per_request import PerRequestBuilder
from clm_replication.polar_agent import ClmHarness
from clm_replication.summary_polar_agent import SummaryHarness
from clm_replication import evaluator

BASE=yaml.safe_load((PREPARED/'C2_TRAIN_CONFIG_BASE.yaml').read_text())

def prepare_config(condition):
    config=deepcopy(BASE)
    config['clm_run_id']=condition.lower()+'-seed1234'
    config['clm_audit_dir']=str(ROOT/'runs'/config['clm_run_id']/'audit')
    config['polar_task_id_template']='clm-'+condition.lower()+'-{rollout_id}-{sample.metadata.qid}'
    config['clm_phase']='training' if condition=='S1' else 'validation'
    config['clm_source_revision']='training-v10-with-summary-prepared-v4'
    config.pop('clm_admission_gate_path',None)
    task=config['polar_task_template']
    task['agent']['import_path']='clm_replication.summary_polar_agent:SummaryHarness'
    volumes=task['runtime']['kwargs']['volumes']
    original=str(ROOT/'agent-bundle-v1/clm_replication')+':'+str(ROOT/'src/clm_replication')+':ro'
    assert volumes.count(original)==1
    volumes[volumes.index(original)]=str(ROOT/'agent-bundle-summary-v4/clm_replication')+':'+str(ROOT/'src/clm_replication')+':ro'
    for path in [ROOT/'env/summary-agent-deps',ROOT/'summary-assets/tokenizer']:
        volumes.append(f'{path}:{path}:ro')
    return config

class SummaryHostTests(unittest.IsolatedAsyncioTestCase):
    def test_instruction_delivery_is_identical(self):
        spec=AgentSpec(import_path='clm_replication.summary_polar_agent:SummaryHarness',model_name='fixture model')
        for instruction in ['Find a public fact.', '中文、多行\nwith `shell` and $HOME', "a'\"; touch /tmp/invalid", 'CLM_INSTRUCTION_JSON\n$(false)']:
            original=ClmHarness(spec).run_steps(instruction)
            modified=SummaryHarness(spec).run_steps(instruction)
            self.assertEqual(len(original),1)
            recovered=modified[0].command.replace(
                'export PYTHONPATH=/mount-data/CLM-Replication/env/summary-agent-deps:', 'export PYTHONPATH=')
            recovered=recovered.replace('-m clm_replication.summary_runner ','-m clm_replication.agent_runner ')
            suffix=' --tokenizer /mount-data/CLM-Replication/summary-assets/tokenizer'
            self.assertTrue(recovered.endswith(suffix))
            self.assertEqual(recovered[:-len(suffix)],original[0].command)

    def test_config_preserves_task_protocol(self):
        for condition in ['S0','S1']:
            config=prepare_config(condition)
            task=config['polar_task_template'];original=BASE['polar_task_template']
            self.assertEqual(task['evaluator'],original['evaluator'])
            self.assertEqual(task['builder'],original['builder'])
            self.assertEqual(task['metadata'],original['metadata'])
            self.assertEqual(task['runtime']['image'],original['runtime']['image'])
            self.assertEqual(task['timeout_seconds'],original['timeout_seconds'])
            self.assertEqual(task['agent']['model_name'],original['agent']['model_name'])
            self.assertNotIn('clm_admission_gate_path',config)
            self.assertTrue(all(Path(path.split(':')[0]).exists() and path.endswith(':ro') for path in task['runtime']['kwargs']['volumes']))

    async def build_fixture(self):
        records=[];events=[]
        for ordinal,command in enumerate(['printf fact',None,'echo POLAR_FINISH_SENTINEL']):
            message={'role':'assistant','content':'CPU fixture summary' if command is None else ''}
            if command is not None:
                message['tool_calls']=[{'id':f'fixture-{ordinal}','type':'function','function':{'name':'bash','arguments':json.dumps({'command':command})}}]
            request={'messages':[{'role':'user','content':'CPU protocol fixture'}]}
            response={'prompt_token_ids':[10,11,12],'choices':[{'token_ids':[20,21],
                'logprobs':{'content':[{'token_id':20,'logprob':-.2},{'token_id':21,'logprob':-.3}]},
                'finish_reason':'stop' if command is None else 'tool_calls','message':message}]}
            records.append(CompletionRecord(completion_id=f'cpu-{ordinal:04d}',request=request,response=response))
            events.append({'ordinal':ordinal,'command':command,'pure_edit':False,
                'submitted':ordinal==2,'call_kind':'summary' if command is None else 'agent',
                'raw_arguments':message['tool_calls'][0]['function']['arguments'] if command else None})
        session=CompletionSession(session_id='cpu-only',completions=records,metadata={'policy_version':0})
        trajectory=await PerRequestBuilder().build(session)
        return trajectory,events

    async def test_summary_is_recorded_and_graded_task_only(self):
        trajectory,events=await self.build_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            session=Path(temporary);(session/'logs/agent').mkdir(parents=True);(session/'artifacts').mkdir()
            (session/'logs/agent/edit_events.json').write_text(json.dumps(events))
            (session/'artifacts/answer.txt').write_text('CPU-only answer')
            instance=evaluator.DeepResearchEvaluator('CPU fixture','CPU fixture')
            with patch.object(evaluator,'grade_answer',return_value={'success':True}) as judge:
                result=await instance.evaluate(trajectory,session_dir=session,session_id='cpu-only',task_id='cpu-fixture',
                    agent_result=AgentRunResult(status='completed',return_code=0))
            self.assertEqual(len(trajectory.traces),3)
            self.assertEqual(trajectory.traces[1].response_messages[0]['content'],'CPU fixture summary')
            self.assertEqual(result.metadata['clm']['pure_edit_by_call'],[False,False,False])
            self.assertEqual(result.outcome_reward,1)
            self.assertTrue(all(trace.loss_mask==[1,1] for trace in trajectory.traces))
            self.assertEqual(judge.call_count,1)
            self.assertEqual(len(result.metadata['clm']['call_costs']),3)

    async def test_event_binding_rejects_fabricated_summary_command(self):
        trajectory,events=await self.build_fixture();events[1]['command']='fabricated'
        with tempfile.TemporaryDirectory() as temporary:
            session=Path(temporary);(session/'logs/agent').mkdir(parents=True)
            (session/'logs/agent/edit_events.json').write_text(json.dumps(events))
            instance=evaluator.DeepResearchEvaluator('CPU fixture','CPU fixture')
            with patch.object(evaluator,'grade_answer') as judge:
                with self.assertRaises(ValueError):
                    await instance.evaluate(trajectory,session_dir=session,session_id='cpu-only',task_id='cpu-fixture',
                        agent_result=AgentRunResult(status='completed',return_code=0))
            judge.assert_not_called()

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SummaryHostTests))
    proof={'accepted':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
           'native_builder_and_evaluator':True,'judge_is_mocked':True,'model_calls':0,'actual_S0_S1_tasks':0,
           'real_Polar_transport_trial':'pending','S1_efficiency_weight_requires_native_actor_env_proof':0}
    (PREPARED/'HOST_CPU_ACTUAL.json').write_text(json.dumps(proof,indent=2))
    if result.wasSuccessful():
        for condition in ['S0','S1']:
            (PREPARED/f'{condition.lower()}-task-config.yaml').write_text(yaml.safe_dump(prepare_config(condition),sort_keys=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
