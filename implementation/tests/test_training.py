"""CPU regressions for the actual Slime loss and dual-channel integration."""

from argparse import Namespace
import copy
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import torch

from clm_replication.advantages import custom_advantage_function
from clm_replication.bridge import post_process_rewards
from polar.rollout.models import SessionResult
from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
from slime_bridge.adapter import session_result_to_samples


class TrainingTests(unittest.TestCase):
    def test_token_mean_partition_and_gradient(self):
        values = torch.tensor([1., 2., 3., 4., 5.], requires_grad=True)
        masks = [torch.tensor([1., 0.]), torch.tensor([1., 1., 1.])]
        with patch('megatron.core.mpu.get_context_parallel_world_size', return_value=1):
            reducer = get_sum_of_sample_mean([4, 5], [2, 3], masks, True)
            whole = reducer(values) / 4
            left = get_sum_of_sample_mean([4], [2], masks[:1], True)(values[:2])
            right = get_sum_of_sample_mean([5], [3], masks[1:], True)(values[2:])
        split = (left + right) / 4
        self.assertTrue(torch.equal(whole, split))
        self.assertTrue(torch.equal(torch.autograd.grad(whole, values, retain_graph=True)[0],
                                    torch.autograd.grad(split, values)[0]))

    def test_dual_channel_exact_and_zero_weight(self):
        batch = {'kl': [torch.zeros(3)], 'rewards': [1.], 'a_eff': [.5], 'role_masks': [[0, 1, 0]]}
        with patch.dict('os.environ', {'POLAR_DUAL_CHANNEL_W_EFF': '0.25'}):
            custom_advantage_function(Namespace(), batch)
        self.assertTrue(torch.equal(batch['advantages'][0], torch.tensor([1., 1.125, 1.])))
        with patch.dict('os.environ', {'POLAR_DUAL_CHANNEL_W_EFF': '0'}):
            custom_advantage_function(Namespace(), batch)
        self.assertTrue(torch.equal(batch['advantages'][0], torch.ones(3)))

    def test_missing_mask_fails(self):
        with self.assertRaises(KeyError):
            custom_advantage_function(Namespace(), {'kl': [torch.zeros(1)], 'rewards': [0.], 'a_eff': [0.]})

    def test_misaligned_mask_fails(self):
        with self.assertRaises(ValueError):
            custom_advantage_function(Namespace(), {
                'kl': [torch.zeros(2)], 'rewards': [1.], 'a_eff': [0.], 'role_masks': [[0]],
            })

    def test_nonbinary_mask_fails(self):
        with self.assertRaises(ValueError):
            custom_advantage_function(Namespace(), {
                'kl': [torch.zeros(1)], 'rewards': [1.], 'a_eff': [0.], 'role_masks': [[2]],
            })

    def test_nonfinite_advantage_fails(self):
        with self.assertRaises(ValueError):
            custom_advantage_function(Namespace(), {
                'kl': [torch.zeros(1)], 'rewards': [1.], 'a_eff': [float('nan')], 'role_masks': [[0]],
            })

    def test_real_trace_conversion_and_trajectory_normalization(self):
        root = Path('/mount-data/CLM-Replication')
        result = json.loads((root / 'management/functional-trajectory-02/RESULT.json').read_text())['results'][0]
        first = SessionResult.model_validate(result)
        second = copy.deepcopy(first)
        second.session_id = 'sk-polar-cpu-fixture-failure'
        second.trajectory.metadata['evaluation']['clm']['success'] = False
        samples = session_result_to_samples(first, 0, trajectory_index=0)
        samples += session_result_to_samples(second, 0, trajectory_index=1)
        with tempfile.TemporaryDirectory(prefix='clm-sample-cpu-') as directory:
            raw, normalized = post_process_rewards(Namespace(clm_audit_dir=directory), samples)
            summary = json.loads((Path(directory) / 'sample-summary-policy-00000.json').read_text())
            self.assertEqual(summary['model_call_samples'], 8)
        self.assertEqual(len(samples), 8)
        self.assertEqual(raw, [1.] * 4 + [0.] * 4)
        self.assertEqual(len(set(round(value, 8) for value in normalized)), 2)
        for sample in samples:
            self.assertEqual(len(sample.metadata['polar']['role_mask']), sample.response_length)
            self.assertEqual(sample.metadata['a_eff'], 0.)


if __name__ == '__main__':
    unittest.main()
