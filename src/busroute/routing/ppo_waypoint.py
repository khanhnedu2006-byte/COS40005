"""Train and run an SB3 policy over budget-safe waypoint actions."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, cast
from zipfile import BadZipFile, ZipFile

import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from busroute.routing.ppo_road import checkpoint_path, fingerprint
from busroute.routing.waypoint_env import (
    ACTION_FEATURES,
    BENCHMARK_CASES,
    Scenario,
    WaypointEnv,
    WaypointFeatures,
    WaypointTrip,
)

if TYPE_CHECKING:
    from Web.types import RouteSnapshot


class WaypointMetadata(TypedDict):
    version: int
    graph_sha256: str
    action_ids: list[str | None]
    steps: int
    requested_steps: int
    seed: int
    threads: int
    training_cases: int
    heldout: list[list[str | int | float]]


class WaypointModelState(Protocol):
    waypoint_metadata: WaypointMetadata


class WaypointPolicy(ActorCriticPolicy):
    """Score each candidate with shared weights and a common action mask."""

    def _build(self, lr_schedule):
        if (
            not isinstance(self.observation_space, spaces.Box)
            or len(self.observation_space.shape) != 1
            or not isinstance(self.action_space, spaces.Discrete)
        ):
            raise ValueError(
                "WaypointPolicy requires a flat Box observation and Discrete action space"
            )
        self.count = int(self.action_space.n)
        self.global_size = self.features_dim - self.count * (ACTION_FEATURES + 1)
        if self.global_size <= 0:
            raise ValueError("Observation space is too small for WaypointPolicy")
        self.action_net = nn.Sequential(
            nn.Linear(self.global_size + ACTION_FEATURES, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.value_net = nn.Sequential(
            nn.Linear(self.features_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        optimizer = cast(Callable[..., torch.optim.Optimizer], self.optimizer_class)
        self.optimizer = optimizer(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def get_distribution(self, obs):
        candidates = obs[:, self.global_size : -self.count].reshape(
            -1,
            self.count,
            ACTION_FEATURES,
        )
        trip = obs[:, None, : self.global_size].expand(-1, self.count, -1)
        logits = self.action_net(torch.cat((trip, candidates), dim=-1)).squeeze(-1)
        mask = obs[:, -self.count :] > 0.5
        # Terminal observations can be all zero. Give them a finite distribution.
        finish = torch.arange(self.count, device=obs.device) == self.count - 1
        valid = torch.where(mask.any(dim=-1, keepdim=True), mask, finish)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        return cast(CategoricalDistribution, self.action_dist).proba_distribution(
            logits
        )

    def predict_values(self, obs):
        return self.value_net(obs)

    def forward(self, obs, deterministic=False):
        distribution = self.get_distribution(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self.predict_values(obs), distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        distribution = self.get_distribution(obs)
        return (
            self.predict_values(obs),
            distribution.log_prob(actions),
            distribution.entropy(),
        )


class TrainingProgress(BaseCallback):
    """Report completed interactions and recent episode rewards."""

    def __init__(self, progress):
        super().__init__()
        self.progress = progress
        self.last_reported = 0

    def _emit(self) -> None:
        episodes = self.model.ep_info_buffer
        reward = sum(ep["r"] for ep in episodes) / len(episodes) if episodes else 0.0
        self.progress({"steps": self.num_timesteps, "mean_recent_reward": reward})
        self.last_reported = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_reported >= 4096:
            self._emit()
        return True

    def _on_training_end(self) -> None:
        if self.num_timesteps > self.last_reported:
            self._emit()


def train(
    features: WaypointFeatures,
    checkpoint: str | Path,
    *,
    steps: int = 100_000,
    seed: int = 0,
    threads: int = 4,
    rollout_steps: int = 256,
    batch_size: int = 64,
    epochs: int = 4,
    progress=None,
) -> PPO:
    """Train fresh weights on verified non-benchmark hub scenarios."""
    if any(
        type(v) is not int or v < 1
        for v in (steps, threads, epochs, rollout_steps, batch_size)
    ) or min(rollout_steps, batch_size) < 2:
        raise ValueError(
            "Waypoint training counts must be positive integers, with batches of at least two"
        )
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a non-negative integer")
    if rollout_steps % batch_size:
        raise ValueError("Batch size must divide waypoint rollout steps")
    path = checkpoint_path(checkpoint)
    torch.set_num_threads(threads)
    scenarios = features.training_cases()
    if progress:
        progress(
            {"training_cases": len(scenarios), "requested_steps": steps, "seed": seed}
        )
    env = WaypointEnv(features, scenarios)
    model = PPO(
        WaypointPolicy,
        env,
        learning_rate=3e-4,
        gamma=1.0,
        gae_lambda=0.95,
        ent_coef=0.01,
        n_steps=rollout_steps,
        batch_size=batch_size,
        n_epochs=epochs,
        seed=seed,
        device="cpu",
    )
    try:
        model.learn(steps, callback=TrainingProgress(progress) if progress else None)
    finally:
        env.close()
    cast(WaypointModelState, model).waypoint_metadata = {
        "version": 1,
        "graph_sha256": fingerprint(features.graph),
        "action_ids": list(features.action_ids),
        "steps": model.num_timesteps,
        "requested_steps": steps,
        "seed": seed,
        "threads": threads,
        "training_cases": len(scenarios),
        "heldout": [list(case) for case in BENCHMARK_CASES],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.zip")
    model.save(temporary)
    temporary.replace(path)
    return model


def load_policy(checkpoint: str | Path, features: WaypointFeatures) -> PPO:
    """Validate checkpoint compatibility before deserializing SB3 weights."""
    path = checkpoint_path(checkpoint)
    try:
        with ZipFile(path) as archive:
            metadata = json.loads(archive.read("data")).get("waypoint_metadata")
    except (BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Expected a trained Waypoint PPO ZIP checkpoint") from exc
    if not isinstance(metadata, dict) or metadata.get("version") != 1:
        raise ValueError("Expected a trained Waypoint PPO ZIP checkpoint")
    if metadata.get("graph_sha256") != fingerprint(features.graph) or metadata.get(
        "action_ids"
    ) != list(features.action_ids):
        raise ValueError(
            "Waypoint PPO checkpoint does not match the graph or action list"
        )
    threads = metadata.get("threads")
    if type(threads) is not int or threads < 1:
        raise ValueError("Invalid threads metadata in Waypoint PPO checkpoint")
    torch.set_num_threads(threads)
    return PPO.load(path, device="cpu")


def infer(features: WaypointFeatures, model: PPO, scenario: Scenario) -> RouteSnapshot:
    """Produce a completed route from the checkpoint's waypoint decisions."""
    trip = WaypointTrip(features, scenario)
    while not trip.done:
        action, _ = model.predict(trip.observe(), deterministic=True)
        trip.step(int(action))
    return trip.snapshot()
