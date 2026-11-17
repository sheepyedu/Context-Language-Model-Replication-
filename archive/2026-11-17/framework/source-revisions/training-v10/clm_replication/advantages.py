"""Validate every dual-channel field before invoking the published CLM hook."""

import math

import torch


def custom_advantage_function(args, rollout_data):
    from slime_bridge.dual_channel_advantage import custom_advantage_function as original

    lengths = [len(value) for value in rollout_data['kl']]
    efficiency = rollout_data['a_eff']
    masks = rollout_data['role_masks']
    rewards = rollout_data['rewards']
    if not (len(lengths) == len(efficiency) == len(masks) == len(rewards)):
        raise ValueError('Dual-channel batch fields are incomplete')
    for length, advantage, reward, mask in zip(lengths, efficiency, rewards, masks, strict=True):
        if not math.isfinite(float(advantage)) or not math.isfinite(float(reward)):
            raise ValueError('Non-finite trajectory advantage')
        if len(mask) != length:
            raise ValueError('Role mask differs from the local generated-token tensor')
        values = mask if torch.is_tensor(mask) else torch.tensor(mask)
        if not bool(torch.all((values == 0) | (values == 1))):
            raise ValueError('Role mask is not binary')
    original(args, rollout_data)
