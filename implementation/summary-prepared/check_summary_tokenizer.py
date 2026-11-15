"""Validate the prepared compactor against the actual frozen Qwen template."""

import asyncio
import hashlib
import json
from pathlib import Path
import tempfile

from transformers import AutoTokenizer
from clm_harness.utils.tool_schemas import BASH_TOOL
from summary_context import SummaryContext


DIRECTORY = Path(__file__).resolve().parent


async def main():
    prompts = DIRECTORY / 'prompts'
    source = json.loads((prompts / 'SOURCE.json').read_bytes())
    for item in source['files']:
        path = prompts / Path(item['path']).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    tokenizer = AutoTokenizer.from_pretrained('/mount-data/Models/Qwen3.5-9B',
                                             local_files_only=True)

    def count_tokens(messages, purpose):
        tools = [BASH_TOOL] if purpose == 'agent' else None
        token_ids = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, add_generation_prompt=True,
            enable_thinking=True, return_dict=False)
        assert isinstance(token_ids, list) and all(isinstance(item, int) for item in token_ids)
        return len(token_ids)

    prefix = [{'role': 'system', 'content': 'Use Bash to solve the task.'},
              {'role': 'user', 'content': 'Preserve the original task. 记录 Unicode。'}]
    lower, upper = 0, 40000
    while lower < upper:
        middle = (lower + upper) // 2
        candidate = prefix + [{'role': 'user', 'content': ' fact' * middle}]
        if count_tokens(candidate, 'agent') < 28672:
            lower = middle + 1
        else:
            upper = middle
    messages = prefix + [{'role': 'user', 'content': ' fact' * lower}]
    requests = []

    async def query(**request):
        requests.append(request)
        assert count_tokens(request['messages'], 'summary') + request['max_tokens'] <= 32768
        return {'choices': [{'finish_reason': 'stop', 'message': {
            'role': 'assistant', 'content': 'The task remains open. Original facts were retained.'}}]}

    with tempfile.TemporaryDirectory() as directory:
        controller = SummaryContext(
            prompt=(prompts / 'prompt.md').read_text(),
            summary_prefix=(prompts / 'summary_prefix.md').read_text(),
            count_tokens=count_tokens, audit_directory=directory)
        retained = await controller.compact_if_needed(messages, query, stage_id=0)
        assert retained[:2] == prefix and len(requests) == 1
        assert count_tokens(retained, 'agent') < 28672
        assert requests[0]['messages'][:-1] == messages
    proof = {'accepted': True, 'scope': 'Actual Qwen tokenizer/template CPU boundary check',
             'before_tokens': count_tokens(messages, 'agent'),
             'after_tokens': count_tokens(retained, 'agent'),
             'trigger': 28672, 'context_limit': 32768, 'task_turn_limit': 100,
             'prompt_source_tree': source['git_tree_sha'], 'new_model_calls': 0,
             'Summary_harness_integrated': False, 'S0_S1_started': False}
    (DIRECTORY / 'TOKENIZER_CPU_ACCEPTED.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
