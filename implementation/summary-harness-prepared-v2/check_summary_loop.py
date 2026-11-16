"""CPU fixtures for Summary call ownership, tool semantics and native Bash."""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

from clm_harness.context_env.env import ContextEnv
from clm_harness.context_env.paths import Paths
from clm_harness.clm_agent import harness
from clm_harness.utils.finish_policy import SUBMIT_MARKER
from clm_replication.agent_runner import ContainerEnvironment
from clm_replication.summary_context import SummaryContext, SummaryOutputError
from clm_replication.summary_loop import SummaryTaskLoop
from clm_replication.summary_runner import make_system_prompt, create_token_counter


def tool_reply(command, *, arguments=None, extra=None):
    calls=[{'id':'tool-call-1','type':'function','function':{'name':'bash','arguments':
            json.dumps({'command':command}) if arguments is None else arguments}}]
    if extra:
        calls.append({'id':'unused-second','type':'function','function':{'name':'bash',
                       'arguments':json.dumps({'command':extra})}})
    return {'id':'cpu-fixture','choices':[{'finish_reason':'tool_calls','message':{
                'role':'assistant','content':'THOUGHT: CPU fixture.','tool_calls':calls}}]}


def text_reply(text='summary', finish='stop'):
    return {'id':'cpu-fixture','choices':[{'finish_reason':finish,'message':{
        'role':'assistant','content':text}}]}


class SummaryLoopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.requests=[]
        self.commands=[]
        self.replies=[tool_reply('echo '+SUBMIT_MARKER)]
        self.error=None
        self.tool_error=None
        self.controller=SummaryContext(prompt='compact',summary_prefix='prior work',
            count_tokens=lambda messages,purpose:sum(len(str(item.get('content','')))+4 for item in messages),
            audit_directory=self.root/'compactions',threshold=1000,context_limit=2000,max_output_tokens=128)
        self.loop=SummaryTaskLoop(controller=self.controller,query=self.query,execute=self.execute,
            format_result=lambda result:result.stdout+'\n(exit_code='+str(result.return_code)+')',
            log_directory=self.root/'logs')
        self.messages=[{'role':'system','content':'system'},{'role':'user','content':'task'}]

    async def query(self,**request):
        self.requests.append(deepcopy(request))
        if self.error:raise self.error
        return deepcopy(self.replies.pop(0))

    async def execute(self,command):
        self.commands.append(command)
        if self.tool_error:raise self.tool_error
        return SimpleNamespace(stdout=SUBMIT_MARKER if command.startswith('echo '+SUBMIT_MARKER) else 'ok',
                               stderr='',return_code=0)

    async def test_bash_submit_and_complete_input_record(self):
        result=await self.loop.run(self.messages)
        self.assertTrue(result['submitted'])
        saved=json.loads((self.root/'logs/calls/0000/INTENT.json').read_text())
        self.assertEqual(saved['messages'],self.messages)
        self.assertEqual(self.loop.events[0]['call_kind'],'agent')
        self.assertTrue(self.loop.events[0]['submitted'])

    async def test_summary_and_task_share_ordered_events(self):
        self.controller.threshold=100
        self.messages.append({'role':'user','content':'x'*100})
        self.replies=[text_reply('facts'),tool_reply('echo '+SUBMIT_MARKER)]
        result=await self.loop.run(self.messages)
        self.assertTrue(result['submitted'])
        self.assertEqual([item['call_kind'] for item in self.loop.events],['summary','agent'])
        self.assertEqual(self.requests[0]['messages'][:-1],self.messages)
        self.assertIsNone(self.requests[0]['tools'])
        self.assertEqual(result['task_steps'],1)
        self.assertTrue(all(not item['pure_edit'] for item in self.loop.events))

    async def test_no_tool_feedback_matches_native_harness(self):
        self.replies=[text_reply('thinking'),tool_reply('echo '+SUBMIT_MARKER)]
        result=await self.loop.run(self.messages)
        self.assertEqual(result['task_steps'],2)
        self.assertIn('(No tool call in your last turn.',self.requests[1]['messages'][-1]['content'])

    async def test_multiple_calls_execute_first_keep_raw_response(self):
        self.replies=[tool_reply('echo '+SUBMIT_MARKER,extra='wrong-command')]
        result=await self.loop.run(self.messages)
        self.assertEqual(len(self.commands),1)
        raw=json.loads((self.root/'logs/calls/0000/RESPONSE.json').read_text())
        self.assertEqual(len(raw['choices'][0]['message']['tool_calls']),2)
        self.assertEqual(len(result['messages'][-2]['tool_calls']),1)

    async def test_malformed_tool_arguments_are_preserved(self):
        self.replies=[tool_reply('',arguments='{broken'),tool_reply('echo '+SUBMIT_MARKER)]
        await self.loop.run(self.messages)
        self.assertEqual(self.loop.events[0]['raw_arguments'],'{broken')
        self.assertIsNone(self.loop.events[0]['command'])

    async def test_context_status_has_matching_tool_return(self):
        response=tool_reply('')
        response['choices'][0]['message']['tool_calls'][0]['function']['name']='context_status'
        self.replies=[response,tool_reply('echo '+SUBMIT_MARKER)]
        await self.loop.run(self.messages)
        self.assertEqual(self.requests[1]['messages'][-1]['role'],'tool')
        self.assertEqual(self.requests[1]['messages'][-1]['tool_call_id'],'tool-call-1')

    async def test_unknown_request_is_consumed_once(self):
        self.error=TimeoutError('CPU fixture unknown')
        with self.assertRaises(TimeoutError):await self.loop.run(self.messages)
        self.assertEqual(len(self.requests),1)
        result=json.loads((self.root/'logs/SUMMARY_LOOP_RESULT.json').read_text())
        self.assertEqual(result['model_calls'],1)
        self.assertEqual(result['disposition'],'error')
        with self.assertRaises(FileExistsError):await self.loop.run(self.messages)
        self.assertEqual(len(self.requests),1)

    async def test_restored_loop_cannot_replay_unknown_episode(self):
        self.error=TimeoutError('CPU fixture unknown')
        with self.assertRaises(TimeoutError):await self.loop.run(self.messages)
        restored=SummaryTaskLoop(controller=self.controller,query=self.query,execute=self.execute,
            format_result=self.loop.format_result,log_directory=self.root/'logs')
        with self.assertRaises(FileExistsError):await restored.run(self.messages)
        self.assertEqual(len(self.requests),1)

    async def test_summary_length_is_known_retained_negative(self):
        self.controller.threshold=100
        self.messages.append({'role':'user','content':'x'*100})
        self.replies=[text_reply('partial','length')]
        with self.assertRaises(SummaryOutputError):await self.loop.run(self.messages)
        self.assertEqual(len(self.requests),1)
        self.assertTrue((self.root/'compactions/0000/REJECTED.json').exists())
        self.assertEqual(len(self.loop.events),1)

    async def test_tool_timeout_retains_original_command(self):
        self.tool_error=TimeoutError('CPU fixture tool timeout')
        with self.assertRaises(TimeoutError):await self.loop.run(self.messages)
        self.assertEqual(self.loop.events[0]['command'],'echo '+SUBMIT_MARKER)
        self.assertEqual(self.loop.events[0]['tool_error'],'TimeoutError')

    async def test_task_step_limit(self):
        self.loop.max_steps=1
        self.replies=[tool_reply('printf ok')]
        result=await self.loop.run(self.messages)
        self.assertEqual(result['disposition'],'task_turn_limit')
        self.assertFalse(result['submitted'])

    async def test_model_call_cap_counts_summary(self):
        self.controller.threshold=100
        self.messages.append({'role':'user','content':'x'*100})
        self.loop.max_model_calls=1
        self.replies=[text_reply('facts')]
        result=await self.loop.run(self.messages)
        self.assertEqual(result['model_calls'],1)
        self.assertEqual(result['task_steps'],0)
        self.assertEqual(result['disposition'],'model_call_limit')

    async def test_unknown_usage_stays_null(self):
        await self.loop.run(self.messages)
        self.assertIsNone(self.loop.events[0]['usage'])
        self.assertIsNone(self.loop.events[0]['output_tokens_per_second'])

    async def test_native_persistent_bash(self):
        shell=ContextEnv(paths=Paths(SimpleNamespace(_CTX_DIR=str(self.root/'ctx'),
            _CTX_FILE=str(self.root/'ctx/unused'),_STATE_DIR=str(self.root/'state'))),
            budget=SimpleNamespace(strict_target=1000))
        environment=ContainerEnvironment(self.root/'native-bash')
        try:
            await shell.setup_dirs(environment)
            first=await shell.execute('mkdir -p "'+str(self.root/'sub')+'" && cd "'+str(self.root/'sub')+'" && export CLM_FIXTURE_VALUE=retained',environment)
            self.assertEqual(first.return_code,0)
            second=await shell.execute('printf "%s\\n" "$CLM_FIXTURE_VALUE"; pwd',environment)
            self.assertEqual(second.stdout,'retained\n'+str(self.root/'sub')+'\n')
            self.assertFalse(Path(shell.paths.ctx_file).exists())
        finally:shell.host_mirror.unlink(missing_ok=True)

    async def test_complete_entrypoint_and_summary_SDK_parameters(self):
        from clm_replication import summary_runner
        instruction=self.root/'instruction.txt'
        instruction.write_text('CPU-only protocol fixture. No benchmark task or model endpoint.')
        captured=[]
        task_calls=0
        summarized=False
        def completion(**parameters):
            nonlocal task_calls,summarized
            captured.append(deepcopy(parameters))
            self.assertEqual(parameters['num_retries'],0)
            self.assertNotIn('tool_choice',parameters)
            self.assertEqual(parameters['temperature'],0.7)
            self.assertEqual(parameters['top_p'],0.95)
            self.assertTrue(parameters['extra_body']['chat_template_kwargs']['enable_thinking'])
            if parameters['tools'] is None:
                summarized=True
                response=text_reply('The CPU fixture printed repeated facts. It can now finish.')
            else:
                task_calls+=1
                command=('echo '+SUBMIT_MARKER if summarized or task_calls>18 else
                         "python3 -c 'print(\" fact\"*2000)'")
                response=tool_reply(command)
            return SimpleNamespace(model_dump=lambda **kwargs:deepcopy(response))
        arguments=['summary_runner','--instruction',str(instruction),'--model','cpu-fixture',
                   '--tokenizer','/mount-data/CLM-Replication/summary-assets/tokenizer']
        with patch.object(sys,'argv',arguments),patch.dict(os.environ,{
                'AGENT_LOG_DIR':str(self.root/'entrypoint'),
                'OPENAI_BASE_URL':'http://invalid.test/v1'}),patch.object(summary_runner.litellm,'completion',completion):
            await summary_runner.main()
        result=json.loads((self.root/'entrypoint/SUMMARY_LOOP_RESULT.json').read_text())
        self.assertTrue(result['submitted'])
        self.assertTrue(summarized)
        self.assertEqual(result['summary_calls'],1)
        self.assertEqual(result['model_calls'],len(captured))
        records=json.loads((self.root/'entrypoint/edit_events.json').read_text())
        self.assertEqual(len(records),len(captured))
        self.assertTrue(all(not item['pure_edit'] for item in records))

    def test_common_system_task_tool_prelude(self):
        prompt=make_system_prompt()
        prefix=harness._SYSTEM_TEMPLATE.partition('## Managing your context')[0].rstrip()
        self.assertTrue(prompt.startswith(prefix))
        self.assertIn('28672',prompt)
        self.assertIn(SUBMIT_MARKER,prompt)
        self.assertNotIn('LIVE_CTX_MAIN',prompt)

    def test_counter_matches_serving_argument_normalization(self):
        count=create_token_counter('/mount-data/CLM-Replication/summary-assets/tokenizer')
        message=tool_reply('printf 你好')['choices'][0]['message']
        messages=self.messages+[message,{'role':'tool','tool_call_id':'tool-call-1','content':'ok'}]
        original=deepcopy(messages)
        normalized=deepcopy(messages)
        normalized[2]['tool_calls'][0]['function']['arguments']=json.loads(message['tool_calls'][0]['function']['arguments'])
        for purpose in ['agent','summary']:
            self.assertEqual(count(messages,purpose),count(normalized,purpose))
        self.assertEqual(messages,original)

    def test_counter_rejects_malformed_retained_arguments(self):
        count=create_token_counter('/mount-data/CLM-Replication/summary-assets/tokenizer')
        message=tool_reply('',arguments='{broken')['choices'][0]['message']
        with self.assertRaises(ValueError):count(self.messages+[message],'agent')

    def test_counter_rejects_nonfinite_tool_arguments(self):
        count=create_token_counter('/mount-data/CLM-Replication/summary-assets/tokenizer')
        for invalid in ['NaN','Infinity','-Infinity']:
            message=tool_reply('',arguments='{"command":'+invalid+'}')['choices'][0]['message']
            with self.assertRaises(ValueError):count(self.messages+[message],'agent')

    def test_actual_agent_tokenizer(self):
        count=create_token_counter('/mount-data/CLM-Replication/summary-assets/tokenizer')
        self.assertGreater(count(self.messages,'agent'),count(self.messages,'summary'))


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(SummaryLoopTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    proof={'accepted':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),
           'errors':len(result.errors),'source':'CPU fixtures with real native Bash and tokenizer',
           'actual_model_generation_calls':0,'actual_S0_S1_tasks':0,'complete_native_Polar_trial':'pending'}
    Path('/work/SUMMARY_LOOP_CPU_ACCEPTED.json').write_text(json.dumps(proof,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
