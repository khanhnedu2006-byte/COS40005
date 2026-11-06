"""Vòng huấn luyện PPO. Khung – nhóm hoàn thiện phần TODO."""

from collections.abc import Callable, Mapping

import torch
from torch import nn

from busroute.routing.env import TourRouteEnv


def compute_gae(
    rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
    gamma: float = 0.99, lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return GAE advantages and value targets for a one-dimensional rollout.

    `rewards`, `values` and boolean `dones` have equal length. The value after
    the final element is assumed zero, including for truncated rollouts.
    """
    adv, last = torch.zeros_like(rewards), 0.0
    for t in reversed(range(len(rewards))):
        next_v = 0.0 if dones[t] or t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * (0.0 if dones[t] else last)
        adv[t] = last
    return adv, adv + values


def ppo_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    adv: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    entropy: torch.Tensor,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> torch.Tensor:
    """Compute the scalar clipped-policy, value, and entropy PPO loss.

    Inputs are batches with compatible shapes. `old_logp`, advantages and
    returns should be detached by the caller when they are fixed targets.
    """
    ratio = torch.exp(new_logp - old_logp)
    policy_loss = -torch.min(
        ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    ).mean()
    value_loss = (returns - values).pow(2).mean()
    return policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()


def train(
    env_factory: Callable[[], TourRouteEnv], encoder: nn.Module, head: nn.Module,
    cfg: Mapping[str, object],
) -> None:
    """Placeholder for the PPO training loop; its intended API may change.

    Args:
        env_factory: Creates a road-graph environment for an episode.
        encoder: Graph encoder to train.
        head: Policy/value head to train.
        cfg: Training configuration.

    Raises:
        NotImplementedError: Always; episode collection and training are TODO.
    """
    # TODO:
    # 1. Mỗi vòng: sinh nhiều episode với (A, B, ngân sách, giờ xuất phát) ngẫu nhiên
    # 2. Lưu (obs, action, logp, reward, value, done)
    # 3. Tính GAE, chuẩn hoá advantage, cập nhật nhiều epoch theo minibatch với ppo_loss
    # 4. Định kỳ đánh giá bằng routing.evaluate và lưu checkpoint vào checkpoints/
    raise NotImplementedError
