"""CPU tests for context replacement, bounded negatives and transport ownership."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from summary_context import SummaryBudgetError, SummaryContext, SummaryOutputError


def count_tokens(messages, purpose):
    return sum(len(message.get('content', '')) + 4 for message in messages)


class SummaryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / 'audit'
        self.controller = SummaryContext(
            prompt='compact', summary_prefix='prior work', count_tokens=count_tokens,
            audit_directory=self.directory, threshold=80, context_limit=150,
            max_output_tokens=40,
        )
        self.messages = [{'role': 'system', 'content': 'system'},
                         {'role': 'user', 'content': 'task'},
                         {'role': 'assistant', 'content': 'h' * 60}]
        self.requests = []
        self.reply = {'choices': [{'finish_reason': 'stop',
                                  'message': {'role': 'assistant', 'content': 'done'}}],
                      'usage': {'prompt_tokens': 93, 'completion_tokens': 1}}

    async def query(self, **request):
        self.requests.append(deepcopy(request))
        return deepcopy(self.reply)

    async def test_below_threshold_has_no_query(self):
        messages = self.messages[:2]
        result = await self.controller.compact_if_needed(messages, self.query, stage_id=0)
        self.assertEqual(result, messages)
        self.assertEqual(self.requests, [])

    async def test_exact_threshold_triggers(self):
        self.controller.threshold = count_tokens(self.messages, 'agent')
        await self.controller.compact_if_needed(self.messages, self.query, stage_id=0)
        self.assertEqual(len(self.requests), 1)

    async def test_protected_prefix_and_complete_history(self):
        before = deepcopy(self.messages)
        retained = await self.controller.compact_if_needed(self.messages, self.query, stage_id=0)
        self.assertEqual(retained[:2], before[:2])
        self.assertEqual(len(retained), 3)
        self.assertEqual(self.requests[0]['messages'][:-1], before)
        self.assertEqual(self.messages, before)
        self.assertIsNone(self.requests[0]['tools'])
        self.assertEqual(self.requests[0]['call_kind'], 'summary')

    async def test_output_cap_respects_remaining_context(self):
        self.controller.context_limit = count_tokens(
            self.messages + [{'role': 'user', 'content': 'compact'}], 'summary') + 5
        await self.controller.compact_if_needed(self.messages, self.query, stage_id=0)
        self.assertEqual(self.requests[0]['max_tokens'], 5)

    async def test_full_input_never_silently_truncates(self):
        self.controller.context_limit = 85
        with self.assertRaises(SummaryBudgetError):
            await self.controller.compact_if_needed(self.messages, self.query, stage_id=0)
        self.assertEqual(self.requests, [])

    async def test_empty_summary_is_retained_rejection(self):
        self.reply['choices'][0]['message']['content'] = ''
        with self.assertRaises(SummaryOutputError):
            await self.controller.compact_if_needed(self.messages, self.query, stage_id=0)
        self.assertTrue((self.directory / '0000/RESPONSE.json').exists())
        self.assertTrue((self.directory / '0000/REJECTED.json').exists())

    async def test_length_is_known_bounded_rejection(self):
        self.reply['choices'][0]['finish_reason'] = 'length'
        with self.assertRaises(SummaryOutputError):
            await self.controller.compact_if_needed(self.messages, self.query, stage_id=0)
        rejection = json.loads((self.directory / '0000/REJECTED.json').read_text())
        self.assertTrue(rejection['request_completed'])
        self.assertEqual(len(self.requests), 1)

    async def test_summary_tool_call_is_never_executed(self):
        self.reply['choices'][0]['message']['tool_calls'] = [{'id': 'unexpected'}]
        with self.assertRaises(SummaryOutputError):
            await self.controller.compact_if_needed(self.messages, self.query, stage_id=0)

    async def test_unknown_query_is_not_retried(self):
        calls = []

        async def timeout(**request):
            calls.append(request)
            raise TimeoutError('transport outcome unknown')

        with self.assertRaises(TimeoutError):
            await self.controller.compact_if_needed(self.messages, timeout, stage_id=0)
        self.assertEqual(len(calls), 1)
        error = json.loads((self.directory / '0000/ERROR.json').read_text())
        self.assertFalse(error['automatic_retry'])

    async def test_large_summary_is_retained_rejection(self):
        self.reply['choices'][0]['message']['content'] = 's' * 100
        with self.assertRaises(SummaryBudgetError):
            await self.controller.compact_if_needed(self.messages, self.query, stage_id=0)
        self.assertTrue((self.directory / '0000/REJECTED.json').exists())

    async def test_repeated_request_keeps_both_original_responses(self):
        await self.controller.compact_if_needed(self.messages, self.query, stage_id=0)
        await self.controller.compact_if_needed(self.messages, self.query, stage_id=1)
        self.assertTrue((self.directory / '0000/RESPONSE.json').exists())
        self.assertTrue((self.directory / '0001/RESPONSE.json').exists())

    async def test_same_stage_cannot_replay_after_unknown(self):
        async def timeout(**request):
            self.requests.append(request)
            raise TimeoutError('unknown')

        with self.assertRaises(TimeoutError):
            await self.controller.compact_if_needed(self.messages, timeout, stage_id=4)
        with self.assertRaises(FileExistsError):
            await self.controller.compact_if_needed(self.messages, self.query, stage_id=4)
        self.assertEqual(len(self.requests), 1)

    async def test_restored_controller_cannot_replay_consumed_stage(self):
        await self.controller.compact_if_needed(self.messages, self.query, stage_id=4)
        restored = SummaryContext(
            prompt='compact', summary_prefix='prior work', count_tokens=count_tokens,
            audit_directory=self.directory, threshold=80, context_limit=150,
            max_output_tokens=40,
        )
        with self.assertRaises(FileExistsError):
            await restored.compact_if_needed(self.messages, self.query, stage_id=4)
        self.assertEqual(len(self.requests), 1)

    async def test_invalid_protected_prefix_fails_before_query(self):
        self.messages[0]['role'] = 'assistant'
        with self.assertRaises(ValueError):
            await self.controller.compact_if_needed(self.messages, self.query, stage_id=0)
        self.assertEqual(self.requests, [])


if __name__ == '__main__':
    unittest.main()
