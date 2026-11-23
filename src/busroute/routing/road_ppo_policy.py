"""Masked full-road-graph policy for Stable-Baselines3 on CPU.

SB3 owns rollout collection, GAE, PPO losses and optimizer updates. This module
only supplies the trainable directed GraphSAGE encoder and masked actor/critic
forward passes. Immutable graph embeddings are reused during a rollout, then
recomputed with gradients for every training minibatch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from gymnasium import spaces
from numpy.typing import NDArray
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn

if TYPE_CHECKING:
    from busroute.routing.road_ppo_env import RoadFeatures

FloatArray = NDArray[np.float32]
IndexArray = NDArray[np.int32]


@dataclass(eq=False)
class StaticRoadGraph:
    """Store O(N+E) graph inputs in policy kwargs, outside rollout observations.

    Attributes:
        first_layer_inputs: Raw self/outgoing/incoming means, shape (N, 18).
        sources: Source node row positions, shape (E,).
        targets: Destination node row positions, shape (E,).
        out_degree: Positive outgoing normalization counts, shape (N, 1).
        in_degree: Positive incoming normalization counts, shape (N, 1).
    """

    first_layer_inputs: FloatArray
    sources: IndexArray
    targets: IndexArray
    out_degree: FloatArray
    in_degree: FloatArray

    @classmethod
    def from_features(cls, features: RoadFeatures) -> StaticRoadGraph:
        """Precompute immutable raw neighbor means without freezing learned layers.

        Args:
            features: Full-graph node attributes and directed edge positions.

        Returns:
            Static inputs used to reconstruct the encoder on checkpoint load.
        """
        outgoing = np.zeros_like(features.x)
        incoming = np.zeros_like(features.x)
        np.add.at(outgoing, features.graph.u, features.x[features.graph.v])
        np.add.at(incoming, features.graph.v, features.x[features.graph.u])
        out_degree = np.maximum(features.degree, 1).astype(np.float32)[:, None]
        in_degree = np.maximum(features.incoming, 1).astype(np.float32)[:, None]
        return cls(
            np.concatenate(
                (features.x, outgoing / out_degree, incoming / in_degree), axis=-1
            ),
            features.graph.u,
            features.graph.v,
            out_degree,
            in_degree,
        )


class RoadEncoder(nn.Module):
    """Encode every road node with two trainable directed GraphSAGE layers.

    Sparse means preserve parallel edges by coalescing their normalized
    contributions. Static tensors are nonpersistent buffers: policy kwargs
    reconstruct them, while the state dict stores learned weights only.
    """

    def __init__(self, graph: StaticRoadGraph, hidden_dim: int) -> None:
        """Build CPU graph buffers and learned layers.

        Args:
            graph: O(N+E) raw attributes, edge indices and normalization counts.
            hidden_dim: Output width for each learned graph layer.
        """
        super().__init__()
        n = len(graph.first_layer_inputs)
        self.register_buffer(
            "inputs", torch.as_tensor(graph.first_layer_inputs), persistent=False
        )
        outgoing = torch.sparse_coo_tensor(
            torch.as_tensor(np.stack((graph.sources, graph.targets)), dtype=torch.long),
            torch.as_tensor(1 / graph.out_degree[graph.sources, 0]),
            (n, n),
            check_invariants=True,
        ).coalesce()
        incoming = torch.sparse_coo_tensor(
            torch.as_tensor(np.stack((graph.targets, graph.sources)), dtype=torch.long),
            torch.as_tensor(1 / graph.in_degree[graph.targets, 0]),
            (n, n),
            check_invariants=True,
        ).coalesce()
        self.register_buffer("outgoing", outgoing, persistent=False)
        self.register_buffer("incoming", incoming, persistent=False)
        self.layer_0 = nn.Linear(18, hidden_dim)
        self.layer_1 = nn.Linear(3 * hidden_dim, hidden_dim)

    def forward(self) -> torch.Tensor:
        """Compute embeddings for the entire graph using current learned weights.

        Returns:
            Differentiable tensor of shape (N, hidden_dim).
        """
        nodes = torch.relu(self.layer_0(self.inputs))
        neighbors = torch.cat(
            (
                nodes,
                torch.sparse.mm(self.outgoing, nodes),
                torch.sparse.mm(self.incoming, nodes),
            ),
            dim=-1,
        )
        return torch.relu(self.layer_1(neighbors))


def decode_observations(
    observations: torch.Tensor,
    k: int,
    n: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Decode a batch of packed road decisions into Torch tensors.

    Args:
        observations: Float32 decisions of shape (B, 2 + 8*K + 5 + P).
        k: Outgoing action-slot width K.
        n: Full road-node count N, used for safe synthetic/terminal indices.

    Returns:
        Current indices, goal indices, candidates, (B,K,6) edge features,
        (B,5+P) trip features and (B,K) boolean masks, in that order.
    """
    indices = observations[:, : 2 + k].long().clamp(0, n - 1)
    edge_end = 2 + k + 6 * k
    edges = observations[:, 2 + k : edge_end].reshape(-1, k, 6)
    return (
        indices[:, 0],
        indices[:, 1],
        indices[:, 2:],
        edges,
        observations[:, edge_end:-k],
        observations[:, -k:] > 0.5,
    )


class RoadPolicy(ActorCriticPolicy):
    """SB3 actor/critic sharing a full-graph encoder and masked edge actions.

    Only static node embeddings are cached, never clock, budget, visited POIs
    or action masks. SB3's train/eval transitions invalidate that cache after
    optimizer updates. Gradient-enabled minibatches always recompute it.

    Attributes:
        encoder: Shared trainable full-graph GraphSAGE network.
        action_net: Per-candidate policy-logit head.
        value_net: Current/goal/trip-state value head.
        graph: Static inputs serialized in policy kwargs.
        k: Number of outgoing action slots, including padding.
        hidden_dim: Graph and head width.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        *,
        graph: StaticRoadGraph,
        k: int,
        hidden_dim: int = 32,
        **kwargs: Any,
    ) -> None:
        """Initialize the standard SB3 policy with road-specific networks.

        Args:
            observation_space: Compact float32 Box produced by RoadFeatures.
            action_space: Discrete outgoing-slot space of size k.
            lr_schedule: SB3 learning-rate schedule.
            graph: Complete immutable graph inputs.
            k: Maximum outgoing degree, without truncating any roads.
            hidden_dim: Width of encoder layers and actor/critic heads.
            **kwargs: Remaining standard ActorCriticPolicy options.

        Raises:
            ValueError: The observation or action spaces are incompatible.
        """
        if (
            not isinstance(observation_space, spaces.Box)
            or len(observation_space.shape) != 1
            or not isinstance(action_space, spaces.Discrete)
            or action_space.n != k
        ):
            raise ValueError(
                "RoadPolicy requires a flat Box and matching Discrete outgoing slots"
            )
        self.graph, self.k, self.hidden_dim = graph, k, hidden_dim
        self.global_dim = int(observation_space.shape[0]) - 2 - 8 * k
        if self.global_dim <= 0:
            raise ValueError("Observation space is too small for RoadPolicy")
        self._cached_embeddings: torch.Tensor | None = None
        kwargs["ortho_init"] = False
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        """Create shared graph weights, both heads and SB3's optimizer.

        Args:
            lr_schedule: Initial and subsequent SB3 learning rates.
        """
        self.encoder = RoadEncoder(self.graph, self.hidden_dim)
        self.action_net = nn.Sequential(
            nn.Linear(3 * self.hidden_dim + 6 + self.global_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.value_net = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + self.global_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        # Concrete SB3 optimizers accept lr, unlike the abstract Optimizer hint.
        optimizer = cast(Callable[..., torch.optim.Optimizer], self.optimizer_class)
        self.optimizer = optimizer(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def clear_embedding_cache(self) -> None:
        """Discard static embeddings after externally changing policy weights."""
        self._cached_embeddings = None

    def train(self, mode: bool = True) -> RoadPolicy:
        """Invalidate cached embeddings when switching PyTorch module mode.

        Args:
            mode: True for training mode; False for evaluation mode.

        Returns:
            This policy instance.
        """
        if mode or mode != self.training:
            self.clear_embedding_cache()
        return super().train(mode)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Load weights and discard any stale cached graph embeddings.

        Args:
            state_dict: Mapping of parameter and persistent buffer tensors.
            strict: Whether keys must match the module state dict exactly.
            assign: Whether to assign items rather than copying into tensors.

        Returns:
            NamedTuple of missing and unexpected keys from PyTorch.
        """
        self.clear_embedding_cache()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def set_training_mode(self, mode: bool) -> None:
        """Invalidate cached embeddings across SB3 optimizer/evaluation phases.

        Args:
            mode: True for learning; False for rollouts or inference.
        """
        if mode or mode != self.training:
            self.clear_embedding_cache()
        super().set_training_mode(mode)

    def _embeddings(self) -> torch.Tensor:
        """Reuse inference-only embeddings or compute a fresh differentiable graph.

        Returns:
            Node embeddings with shape (N, hidden_dim). No gradient-bearing
            tensor is ever retained between training minibatches.
        """
        if self.training or torch.is_grad_enabled():
            return self.encoder()
        if self._cached_embeddings is None:
            self._cached_embeddings = self.encoder()
        return self._cached_embeddings

    def _inputs(
        self, obs: PyTorchObs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct actor/critic inputs using one shared encoder pass.

        Args:
            obs: Batched Box observations; Dict observations are unsupported.

        Returns:
            Per-edge actor inputs, per-state critic inputs and action masks.
        """
        observations = cast(torch.Tensor, obs)
        nodes = self._embeddings()
        current, target, candidates, edges, globals, mask = decode_observations(
            observations,
            self.k,
            len(self.graph.first_layer_inputs),
        )
        shape = (observations.shape[0], self.k, self.hidden_dim)
        actor_inputs = torch.cat(
            (
                nodes[current, None].expand(shape),
                nodes[target, None].expand(shape),
                nodes[candidates],
                edges,
                globals[:, None].expand(-1, self.k, -1),
            ),
            dim=-1,
        )
        critic_inputs = torch.cat((nodes[current], nodes[target], globals), dim=-1)
        return actor_inputs, critic_inputs, mask

    def _distribution(
        self, inputs: torch.Tensor, mask: torch.Tensor
    ) -> CategoricalDistribution:
        """Build the masked distribution used by every SB3 action path.

        Args:
            inputs: Per-candidate actor features, shape (B,K,F).
            mask: Boolean validity flags, shape (B,K).

        Returns:
            Categorical distribution with zero probability on invalid slots.
            All-masked synthetic samples use slot zero as a finite sentinel;
            real all-masked routing states are rejected before prediction.
        """
        logits = self.action_net(inputs).squeeze(-1)
        sentinel = torch.arange(self.k, device=mask.device)[None, :] == 0
        valid = torch.where(mask.any(dim=-1, keepdim=True), mask, sentinel)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        return cast(CategoricalDistribution, self.action_dist).proba_distribution(
            action_logits=logits
        )

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return actions, values and log probabilities for SB3 rollouts.

        Args:
            obs: Batched packed road observations.
            deterministic: Select the action mode instead of sampling.

        Returns:
            Action indices (B,), values (B,1) and log probabilities (B,).
        """
        actor, critic, mask = self._inputs(obs)
        distribution = self._distribution(actor, mask)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self.value_net(critic), distribution.log_prob(actions)

    def evaluate_actions(
        self, obs: PyTorchObs, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Evaluate stored actions with fresh gradient-bearing graph embeddings.

        Args:
            obs: Packed minibatch observations from SB3's rollout buffer.
            actions: Corresponding outgoing-slot indices.

        Returns:
            Values, action log probabilities and policy entropy for PPO losses.
        """
        actor, critic, mask = self._inputs(obs)
        distribution = self._distribution(actor, mask)
        return (
            self.value_net(critic),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def get_distribution(self, obs: PyTorchObs) -> CategoricalDistribution:
        """Return masked probabilities for SB3's predict API.

        Args:
            obs: Batched packed Box observations.

        Returns:
            Categorical distribution over outgoing edge slots.
        """
        actor, _, mask = self._inputs(obs)
        return self._distribution(actor, mask)

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        """Predict critic values for rollout-cut bootstrapping.

        Args:
            obs: Batched packed Box observations.

        Returns:
            Value tensor of shape (B,1).
        """
        _, critic, _ = self._inputs(obs)
        return self.value_net(critic)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        """Preserve graph arguments when saving the policy independently.

        Returns:
            Standard SB3 constructor settings plus road graph dimensions.
        """
        return {
            **super()._get_constructor_parameters(),
            "graph": self.graph,
            "k": self.k,
            "hidden_dim": self.hidden_dim,
        }
