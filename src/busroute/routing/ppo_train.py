"""Legacy teaching helpers, retained outside the active library-based pipelines.

For training, use ``ppo_road.train`` (complete road graph) or ``ppo_od.train``
(OD matrix). They delegate PPO updates and GAE to SB3 and SBX, respectively. The functions below
illustrate the older Torch mathematics and are not used by those pipelines.
"""

from collections.abc import Callable, Mapping

import torch
from torch import nn

from busroute.routing.env import TourRouteEnv


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Illustrate GAE advantages and value targets for a legacy Torch rollout.

    Args:
        rewards: One-dimensional reward tensor with T entries.
        values: Predicted values for those same T states.
        dones: Boolean terminal flags of length T.
        gamma: Discount factor for future rewards.
        lam: GAE bias/variance parameter.

    Returns:
        Advantage and value-target tensors of length T. This teaching helper
        assumes zero value after the last entry, even at a nonterminal cut;
        active library-based training uses correctly bootstrapped rollout buffers.
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
    """Illustrate the legacy Torch clipped-policy, value and entropy loss.

    Args:
        new_logp: Current policy log probabilities for sampled actions.
        old_logp: Detached behavior-policy log probabilities for those actions.
        adv: Detached advantage estimates with compatible batch shape.
        returns: Detached value targets with compatible batch shape.
        values: Current critic estimates.
        entropy: Current policy entropy per sample.
        clip_eps: Policy-ratio clipping range.
        vf_coef: Squared-value-loss coefficient.
        ent_coef: Entropy regularization coefficient.

    Returns:
        Scalar differentiable combined loss. This teaching helper is not used
        by either active library-based training pipeline.
    """
    ratio = torch.exp(new_logp - old_logp)
    policy_loss = -torch.min(
        ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    ).mean()
    value_loss = (returns - values).pow(2).mean()
    return policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()


def train(
    env_factory: Callable[[], TourRouteEnv],
    encoder: nn.Module,
    head: nn.Module,
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
