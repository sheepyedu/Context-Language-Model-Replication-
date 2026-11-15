"""Codex-prompt compaction for the separately prepared Summary conditions."""

from copy import deepcopy
import json
from pathlib import Path


class SummaryBudgetError(RuntimeError):
    """The complete compaction input or resulting context cannot fit."""


class SummaryOutputError(RuntimeError):
    """A completed request did not produce a usable complete text summary."""


class SummaryContext:
    """Preserve the original system/task and compact only the accumulated history.

    The RL threshold and task-turn limit follow paper Appendix E. The caller
    records every request through the ordinary rollout transport. This class
    never executes tools or retries a request with an unknown outcome.
    """

    task_turn_limit = 100

    def __init__(self, *, prompt, summary_prefix, count_tokens, audit_directory,
                 threshold=28672, context_limit=32768, max_output_tokens=4096):
        if not 0 < threshold < context_limit or max_output_tokens < 1:
            raise ValueError('Invalid Summary context limits')
        if not prompt.strip() or not summary_prefix.strip():
            raise ValueError('Both frozen Codex prompt artifacts are required')
        self.prompt = prompt
        self.summary_prefix = summary_prefix
        self.count_tokens = count_tokens
        self.audit_directory = Path(audit_directory)
        self.threshold = threshold
        self.context_limit = context_limit
        self.max_output_tokens = max_output_tokens
        self.compaction_count = 0

    @staticmethod
    def _validate_prefix(messages):
        if len(messages) < 2:
            raise ValueError('The system and original task messages are required')
        if messages[0].get('role') != 'system' or messages[1].get('role') != 'user':
            raise ValueError('The protected prefix must be system followed by task')

    async def compact_if_needed(self, messages, query, *, stage_id):
        """Return the original history or one completed, recorded compaction.

        ``count_tokens(messages, purpose)`` must use the actual serving template,
        including tools for ``agent`` and excluding them for ``summary``.
        ``query`` accepts messages, max_tokens, tools and call_kind. It must return
        the unmodified Chat Completions response and persist transport evidence.
        """
        if not isinstance(stage_id, int) or isinstance(stage_id, bool) or stage_id < 0:
            raise ValueError('A stable task-stage identifier is required')
        self._validate_prefix(messages)
        original = deepcopy(messages)
        before_tokens = self.count_tokens(original, 'agent')
        if before_tokens < self.threshold:
            return original
        request_messages = original + [{'role': 'user', 'content': self.prompt}]
        prompt_tokens = self.count_tokens(request_messages, 'summary')
        output_cap = min(self.max_output_tokens, self.context_limit - prompt_tokens)
        if output_cap < 1:
            raise SummaryBudgetError('The complete compaction input exceeds the serving context')
        directory = self.audit_directory / f'{stage_id:04d}'
        directory.mkdir(parents=True, exist_ok=False)
        self.compaction_count += 1
        request = {'messages': request_messages, 'tools': None,
                   'max_tokens': output_cap, 'call_kind': 'summary'}
        (directory / 'INTENT.json').write_text(json.dumps({
            'request': request, 'before_tokens': before_tokens,
            'compaction_prompt_tokens': prompt_tokens, 'stage_id': stage_id,
        }, ensure_ascii=False, indent=2))
        try:
            response = await query(**deepcopy(request))
        except Exception as error:
            (directory / 'ERROR.json').write_text(json.dumps({
                'error_type': type(error).__name__, 'automatic_retry': False,
                'original_history_retained': True,
            }, indent=2))
            raise
        (directory / 'RESPONSE.json').write_text(json.dumps(response, ensure_ascii=False, indent=2))
        try:
            choices = response.get('choices') or []
            if len(choices) != 1 or choices[0].get('finish_reason') != 'stop':
                raise SummaryOutputError('Compaction requires one completed text choice')
            message = choices[0].get('message') or {}
            summary = message.get('content')
            if not isinstance(summary, str) or not summary.strip() or message.get('tool_calls'):
                raise SummaryOutputError('Compaction produced empty text or a tool call')
            retained = original[:2] + [{
                'role': 'user', 'content': self.summary_prefix + '\n\n' + summary,
            }]
            after_tokens = self.count_tokens(retained, 'agent')
            if after_tokens >= self.threshold:
                raise SummaryBudgetError('The completed summary does not fit below the trigger')
        except (SummaryOutputError, SummaryBudgetError) as error:
            (directory / 'REJECTED.json').write_text(json.dumps({
                'error_type': type(error).__name__, 'reason': str(error),
                'request_completed': True, 'original_history_retained': True,
                'automatic_retry': False,
            }, indent=2))
            raise
        (directory / 'ACCEPTED.json').write_text(json.dumps({
            'protected_prefix_unchanged': retained[:2] == original[:2],
            'before_tokens': before_tokens, 'after_tokens': after_tokens,
            'complete_original_history_sent': True, 'automatic_retry': False,
            'required_summary_credit': 'task_only', 'efficiency_role_mask': 0,
        }, indent=2))
        return retained
