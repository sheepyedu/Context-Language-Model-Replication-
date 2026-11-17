"""Trajectory-level rewards and prefix-reuse cost from the CLM recipe."""

from dataclasses import dataclass
import math
import statistics


@dataclass(frozen=True)
class Outcome:
    trajectory_id: str
    success: bool
    clean_finish: bool
    flops: float
    no_tool: bool = False
    format_failure: bool = False

    @property
    def reward(self):
        if self.success:
            return 1.0
        return -0.3 * self.no_tool - 0.2 * self.format_failure


def group_advantages(outcomes, efficiency_weight=0.25):
    """Compute each complete trajectory once, before any trace expansion."""
    if len(outcomes) < 2:
        raise ValueError("A rollout group requires at least two trajectories")
    ids = [item.trajectory_id for item in outcomes]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate trajectory in rollout group")
    if not math.isfinite(efficiency_weight) or efficiency_weight < 0:
        raise ValueError("Invalid efficiency weight")
    for item in outcomes:
        if not math.isfinite(item.flops) or item.flops <= 0:
            raise ValueError("Missing or invalid cost cannot be replaced with zero")
    rewards = [item.reward for item in outcomes]
    mean_reward = statistics.mean(rewards)
    std_reward = statistics.stdev(rewards)
    successes = [item for item in outcomes if item.success and item.clean_finish]
    mean_cost = statistics.mean(item.flops for item in successes) if len(successes) >= 2 else None
    result = {}
    for item in outcomes:
        efficiency = 0.0
        if mean_cost is not None and item.success and item.clean_finish:
            efficiency = max(-1.0, min(1.0, (mean_cost - item.flops) / mean_cost))
        result[item.trajectory_id] = {
            "reward": item.reward,
            "a_out": (item.reward - mean_reward) / (std_reward + 1e-6),
            "a_eff": efficiency,
            "efficiency_weight": efficiency_weight,
        }
    return {
        "accepted": std_reward > 0.0,
        "reason": None if std_reward > 0.0 else "zero_task_reward_variance",
        "successful_clean_count": len(successes),
        "mean_success_flops": mean_cost,
        "trajectories": result,
    }


def prefix_reuse_flops(calls, c_token=13_839_200_000.0, c_attn=131_072.0):
    """Use the immediately preceding original input, including rolled-back calls."""
    previous = []
    total = 0.0
    rows = []
    for call in calls:
        prompt = call["prompt_ids"]
        response = call["response_ids"]
        if not prompt or not response:
            raise ValueError("Complete original prompt and response IDs required")
        reused = 0
        for left, right in zip(previous, prompt):
            if left != right:
                break
            reused += 1
        p, g = len(prompt), len(response)
        cost = c_token * (p - reused + g) + c_attn * (
            (p * p - reused * reused) / 2 + g * p + g * g / 2
        )
        rows.append({"prompt": p, "generated": g, "reused": reused, "flops": cost})
        total += cost
        previous = prompt
    if not rows:
        raise ValueError("Cannot compute cost of an empty trajectory")
    return total, rows


def validate_masks(response_ids, logprobs, loss_mask, role_mask):
    size = len(response_ids)
    if not size or any(len(items) != size for items in (logprobs, loss_mask, role_mask)):
        raise ValueError("Response, logprob and mask lengths must match exactly")
    if any(not math.isfinite(value) for value in logprobs):
        raise ValueError("Non-finite rollout logprob")
    if any(value not in (0, 1) for value in loss_mask + role_mask):
        raise ValueError("Masks must be binary")
    if any(role > loss for role, loss in zip(role_mask, loss_mask)):
        raise ValueError("Efficiency mask includes a non-generated token")
