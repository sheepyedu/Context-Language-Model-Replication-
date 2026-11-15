import math
import unittest

from clm_replication.rewards import Outcome, group_advantages, prefix_reuse_flops, validate_masks


class RewardsTest(unittest.TestCase):
    def test_efficiency_only_reranks_successes(self):
        result = group_advantages([Outcome("a", True, True, 10), Outcome("b", True, True, 30), Outcome("c", False, True, 1)])
        self.assertEqual(result["trajectories"]["a"]["a_eff"], 0.5)
        self.assertEqual(result["trajectories"]["b"]["a_eff"], -0.5)
        self.assertEqual(result["trajectories"]["c"]["a_eff"], 0)

    def test_one_success_has_no_efficiency(self):
        result = group_advantages([Outcome("a", True, True, 10), Outcome("b", False, True, 30)])
        self.assertTrue(all(row["a_eff"] == 0 for row in result["trajectories"].values()))

    def test_penalties_apply_once_to_failure(self):
        self.assertEqual(Outcome("a", False, True, 1, True, True).reward, -0.5)
        self.assertEqual(Outcome("b", True, True, 1, True, True).reward, 1)

    def test_all_success_group_filtered_even_with_efficiency(self):
        self.assertFalse(group_advantages([Outcome("a", True, True, 10), Outcome("b", True, True, 30)])["accepted"])

    def test_failed_clean_status_does_not_receive_efficiency(self):
        result = group_advantages([Outcome("a", True, False, 10), Outcome("b", True, True, 30), Outcome("c", False, True, 1)])
        self.assertTrue(all(row["a_eff"] == 0 for row in result["trajectories"].values()))

    def test_reject_duplicate_trajectory(self):
        with self.assertRaises(ValueError):
            group_advantages([Outcome("a", True, True, 1), Outcome("a", False, True, 2)])

    def test_reject_missing_or_nonfinite_cost(self):
        for cost in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                group_advantages([Outcome("a", True, True, cost), Outcome("b", False, True, 1)])

    def test_cost_uses_previous_input(self):
        total, rows = prefix_reuse_flops([
            {"prompt_ids": [1, 2, 3], "response_ids": [4]},
            {"prompt_ids": [1, 2, 3, 4, 5], "response_ids": [6]},
            {"prompt_ids": [1, 7], "response_ids": [8]},
        ], c_token=1, c_attn=0)
        self.assertEqual([row["reused"] for row in rows], [0, 3, 1])
        self.assertEqual(total, 9)

    def test_mask_alignment(self):
        validate_masks([1, 2], [-1.0, -0.5], [1, 1], [0, 1])
        with self.assertRaises(ValueError):
            validate_masks([1, 2], [-1.0], [1, 1], [0, 1])

    def test_mask_must_only_reach_generated_positions(self):
        with self.assertRaises(ValueError):
            validate_masks([1, 2], [-1.0, -0.5], [1, 0], [0, 1])


if __name__ == "__main__":
    unittest.main()
