"""Prepared Summary entrypoint using the existing native Bash and transport."""
import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import litellm
from transformers import AutoTokenizer
from clm_harness.clm_agent import harness
from clm_harness.context_env.env import ContextEnv
from clm_harness.context_env.paths import Paths
from clm_harness.task_templates import instance_template
from clm_harness.utils.finish_policy import FinishPolicy
from clm_harness.utils.lm_compat import token_cap_key
from clm_harness.utils.tool_schemas import BASH_TOOL

from clm_replication.agent_runner import ContainerEnvironment
from clm_replication.summary_context import SummaryContext
from clm_replication.summary_loop import SummaryTaskLoop


def create_token_counter(directory):
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True,
                                               trust_remote_code=False)
    def reject_nonfinite_constant(value):
        raise ValueError('Non-finite JSON constant in tool arguments: ' + value)

    def count_tokens(messages, purpose):
        renderable = deepcopy(messages)
        # Mirror SGLang's OpenAI serving normalization for the local token counter.
        for message in renderable:
            if message.get('role') == 'assistant' and isinstance(message.get('tool_calls'), list):
                for call in message['tool_calls']:
                    function = call['function']
                    if isinstance(function.get('arguments'), str):
                        function['arguments'] = json.loads(
                            function['arguments'], parse_constant=reject_nonfinite_constant)
        values = tokenizer.apply_chat_template(renderable, tools=[BASH_TOOL] if purpose=='agent' else None,
            tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False)
        if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
            raise TypeError('The serving template must return integer token IDs')
        return len(values)
    return count_tokens


def make_system_prompt():
    prefix, marker, _ = harness._SYSTEM_TEMPLATE.partition('## Managing your context')
    if not marker:
        raise ValueError('The pinned common task/tool prelude was not found')
    return (prefix.rstrip() + '\n\nEarlier interaction history is automatically summarized '
            'when the context reaches 28672 tokens.\n\n' + FinishPolicy('submit').prompt_fragment())


async def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--instruction',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--tokenizer',required=True)
    args=parser.parse_args()
    logs=Path(os.environ['AGENT_LOG_DIR'])
    logs.mkdir(parents=True,exist_ok=True)
    prompts=Path(__file__).parent/'summary_prompts'
    source=json.loads((prompts/'SOURCE.json').read_text())
    for item in source['files']:
        assert hashlib.sha256((prompts/Path(item['path']).name).read_bytes()).hexdigest()==item['sha256']
    counter=create_token_counter(args.tokenizer)
    controller=SummaryContext(prompt=(prompts/'prompt.md').read_text(),
        summary_prefix=(prompts/'summary_prefix.md').read_text(),count_tokens=counter,
        audit_directory=logs/'compactions')
    environment=ContainerEnvironment(logs)
    shell=ContextEnv(paths=Paths(SimpleNamespace(_CTX_DIR='/tmp/.summary_state',
            _CTX_FILE='/tmp/.summary_state/unused.txt',_STATE_DIR='/tmp/.bash_ctx_state')),
            budget=SimpleNamespace(strict_target=28672),
            command_timeout=120,observation_max_chars=10000,obs_cfg=harness._OBS_CFG)
    await shell.setup_dirs(environment)

    async def query(*,messages,tools,max_tokens,call_kind):
        model='openai/'+args.model
        parameters=dict(model=model,messages=messages,tools=tools,api_base=os.environ['OPENAI_BASE_URL'],
                        temperature=0.7,top_p=0.95,num_retries=0,drop_params=True,
                        timeout=600,request_timeout=600,
                        extra_body={'chat_template_kwargs':{'enable_thinking':True}})
        parameters[token_cap_key(model)]=max_tokens
        response=await asyncio.wait_for(asyncio.to_thread(litellm.completion,**parameters),timeout=630)
        return response.model_dump(mode='json')

    async def execute(command):
        return await shell.execute(command,environment)

    def format_result(result):
        return shell.format_tool_result(result)

    messages=[{'role':'system','content':make_system_prompt()},
              {'role':'user','content':instance_template(None).replace('{{task}}',Path(args.instruction).read_text().strip())}]
    (logs/'SUMMARY_EFFECTIVE_CONFIG.json').write_text(json.dumps({
        'method':'Summary','system_prompt':messages[0]['content'],'context_threshold':28672,
        'server_context_limit':32768,'max_output_tokens':4096,'max_task_steps':100,
        'max_model_calls':224,'temperature':0.7,'top_p':0.95,'enable_thinking':True,
        'summary_prompt_source':source,'tokenizer':args.tokenizer,
        'implementation_default':'Pinned CLM task/tool prelude plus automatic-summary notice',
    },ensure_ascii=False,indent=2))
    loop=SummaryTaskLoop(controller=controller,query=query,execute=execute,
                         format_result=format_result,log_directory=logs)
    try:
        await loop.run(messages)
    finally:
        shell.host_mirror.unlink(missing_ok=True)


if __name__=='__main__':
    asyncio.run(main())
