import unittest

from clm_replication.accounting import aggregate, token_counts


class AccountingTests(unittest.TestCase):
    def test_cache_and_thinking_are_subsets(self):
        row = token_counts({'usage': {
            'prompt_tokens': 100, 'completion_tokens': 30, 'total_tokens': 130,
            'prompt_tokens_details': {'cached_tokens': 70},
            'completion_tokens_details': {'reasoning_tokens': 20},
        }})
        self.assertEqual(row['total_tokens'], 130)
        self.assertEqual(row['cached_input_tokens_subset'], 70)
        self.assertEqual(row['reasoning_output_tokens_subset'], 20)

    def test_ids_are_explicit_fallback(self):
        row = token_counts({'choices': [{'input_token_ids': [1, 2], 'token_ids': [3, 4, 5]}]})
        self.assertEqual(row['total_tokens'], 5)
        self.assertEqual(row['input_count_source'], 'token_ids')

    def test_missing_usage_stays_unknown(self):
        row = token_counts(None)
        summary = aggregate([row])
        self.assertIsNone(summary['complete_total_tokens'])
        self.assertEqual(summary['missing_total_tokens_calls'], 1)

    def test_zero_usage_is_known(self):
        row = token_counts({'usage': {'prompt_tokens': 0, 'completion_tokens': 0}})
        self.assertEqual(aggregate([row])['complete_total_tokens'], 0)

    def test_judge_and_rollout_both_contribute(self):
        first = token_counts({'usage': {'prompt_tokens': 100, 'completion_tokens': 30}})
        second = token_counts({'usage': {'prompt_tokens': 40, 'completion_tokens': 10}})
        self.assertEqual(aggregate([first, second])['complete_total_tokens'], 180)


if __name__ == '__main__':
    unittest.main()
