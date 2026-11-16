"""Bash task loop with recorded automatic history compaction."""
from copy import deepcopy
import json
from pathlib import Path
import time

from clm_harness.clm_agent.harness import _extract_tool_call
from clm_harness.context_env.env import ContextEnv
from clm_harness.utils.finish_policy import FinishPolicy, SUBMIT_MARKER
from clm_harness.utils.tool_schemas import BASH_TOOL


class SummaryTaskLoop:
    """Use the ordinary request transport for both task and summary calls."""

    def __init__(self, *, controller, query, execute, format_result, log_directory,
                 max_steps=100, max_model_calls=224):
        self.controller = controller
        self.query = query
        self.execute = execute
        self.format_result = format_result
        self.logs = Path(log_directory)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.max_model_calls = max_model_calls
        self.events = []
        self.model_requests = 0
        self.summary_requests = 0
        self.finish_policy = FinishPolicy('submit')

    def save_events(self):
        temporary = self.logs / 'edit_events.tmp'
        temporary.write_text(json.dumps(self.events, ensure_ascii=False, indent=2))
        temporary.replace(self.logs / 'edit_events.json')

    async def recorded_query(self, **request):
        if self.model_requests >= self.max_model_calls:
            raise RuntimeError('The Summary model-call budget is exhausted')
        ordinal = self.model_requests
        directory = self.logs / 'calls' / f'{ordinal:04d}'
        directory.mkdir(parents=True, exist_ok=False)
        self.model_requests += 1
        self.summary_requests += int(request['call_kind'] == 'summary')
        (directory / 'INTENT.json').write_text(json.dumps(request, ensure_ascii=False, indent=2))
        started = time.monotonic()
        try:
            response = await self.query(**deepcopy(request))
        except Exception as error:
            (directory / 'ERROR.json').write_text(json.dumps({
                'error_type': type(error).__name__, 'automatic_retry': False}))
            raise
        (directory / 'RESPONSE.json').write_text(json.dumps(response, ensure_ascii=False, indent=2))
        choices = response.get('choices') or []
        if len(choices) != 1 or not isinstance(choices[0].get('message'), dict):
            raise ValueError('A recorded completion requires one message choice')
        message = deepcopy(choices[0]['message'])
        action = _extract_tool_call(message)
        tools = choices[0]['message'].get('tool_calls') or []
        elapsed = time.monotonic() - started
        usage = response.get('usage') or {}
        output_tokens = usage.get('completion_tokens')
        self.events.append({
            'ordinal': ordinal, 'response_id': response.get('id'),
            'call_kind': request['call_kind'], 'pure_edit': False,
            'command': action.get('command'), 'submitted': False,
            'raw_arguments': tools[0]['function'].get('arguments') if tools else None,
            'elapsed_seconds': elapsed, 'usage': response.get('usage'),
            'output_tokens_per_second': (output_tokens / elapsed
                                         if isinstance(output_tokens, int) and not isinstance(output_tokens, bool) and elapsed > 0 else None),
        })
        self.save_events()
        return response

    async def run(self, initial_messages):
        messages = deepcopy(initial_messages)
        self.controller._validate_prefix(messages)
        with (self.logs / 'SUMMARY_RUN_INTENT.json').open('x') as handle:
            json.dump({'initial_messages':messages, 'max_steps':self.max_steps,
                       'max_model_calls':self.max_model_calls},handle,ensure_ascii=False,indent=2)
        submitted = False
        task_steps = 0
        disposition = 'task_turn_limit'
        self.save_events()
        try:
            for step in range(self.max_steps):
                if self.model_requests >= self.max_model_calls:
                    disposition = 'model_call_limit'
                    break
                messages = await self.controller.compact_if_needed(
                    messages, self.recorded_query, stage_id=step)
                if self.model_requests >= self.max_model_calls:
                    disposition = 'model_call_limit'
                    break
                prompt_tokens = self.controller.count_tokens(messages, 'agent')
                output_cap = min(self.controller.max_output_tokens,
                                 self.controller.context_limit - prompt_tokens)
                if output_cap < 1:
                    raise ValueError('The complete task request cannot fit the context limit')
                response = await self.recorded_query(messages=messages, tools=[BASH_TOOL],
                                                       max_tokens=output_cap, call_kind='agent')
                message = deepcopy(response['choices'][0]['message'])
                message['content'] = message.get('content') or ''
                action = _extract_tool_call(message)
                event = self.events[-1]
                task_steps += 1
                if action['type'] == 'no_tool_call':
                    message.pop('tool_calls', None)
                    messages.extend([message, {'role': 'user', 'content':
                        '(No tool call in your last turn. When ready, issue your next `bash` command to continue the task.)'}])
                    continue
                if action['type'] == 'context_status':
                    messages.extend([message, {'role': 'tool', 'tool_call_id':action.get('tool_call_id') or '',
                        'content': f'{prompt_tokens}/{self.controller.threshold} tokens in context.'}])
                    continue
                command = action.get('command') or ''
                try:
                    result = await self.execute(command)
                except Exception as error:
                    event['tool_error'] = type(error).__name__
                    self.save_events()
                    raise
                content = ContextEnv.append_submit_warning(command, self.format_result(result))
                submitted = (self.finish_policy.is_finish_command(command)
                             and result.return_code == 0 and SUBMIT_MARKER in content)
                event.update(return_code=result.return_code, ctx_changed=False, submitted=submitted)
                self.save_events()
                messages.extend([message, {'role':'tool','tool_call_id':action.get('tool_call_id') or '',
                                           'content':content}])
                if submitted:
                    disposition = 'submitted'
                    break
        except Exception as error:
            disposition = 'error'
            (self.logs / 'SUMMARY_LOOP_ERROR.json').write_text(json.dumps({
                'error_type':type(error).__name__, 'automatic_retry':False}))
            raise
        finally:
            self.save_events()
            (self.logs / 'trajectory.json').write_text(json.dumps(messages, ensure_ascii=False, indent=2))
            (self.logs / 'SUMMARY_LOOP_RESULT.json').write_text(json.dumps({
                'submitted': submitted, 'task_steps':task_steps, 'model_calls':self.model_requests,
                'summary_calls':self.summary_requests, 'usable_completion_events':len(self.events),
                'max_steps':self.max_steps, 'max_model_calls':self.max_model_calls,
                'disposition':disposition,
            }, indent=2))
        return {'submitted':submitted,'task_steps':task_steps,'model_calls':self.model_requests,
                'disposition':disposition,'messages':messages}
