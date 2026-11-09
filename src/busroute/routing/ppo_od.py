"""SBX PPO with a masked GraphSAGE policy for OD-matrix routing."""

from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import flax.linen as nn
import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from gymnasium import spaces
from sbx import PPO
from sbx.ppo.policies import Actor, Critic

from busroute.routing.od_env import ODRouteEnv
from busroute.routing.metaheuristics import _Search
from busroute.routing.orienteering import ODMatrix


@dataclass
class PPOConfig:
    total_steps: int = 200_000
    lr: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    hidden_dim: int = 64
    rollout_steps: int = 256
    epochs: int = 4
    minibatch_size: int = 32
    seed: int = 0
    vf_coef: float = 0.5
    ent_coef: float = 0.01


def matrix_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_coordinates(path: str | Path) -> dict[str, tuple[float, float]]:
    coords = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            for prefix in ("from", "to"):
                point = row[prefix + "_id"]
                coord = (float(row[prefix + "_lat"]), float(row[prefix + "_lon"]))
                if point in coords and coords[point] != coord:
                    raise ValueError(f"Inconsistent coordinates for {point}")
                coords[point] = coord
    return coords


def scenarios(
    matrix: ODMatrix, count: int, seed: int
) -> list[tuple[str, str, int, float]]:
    if count <= 0:
        raise ValueError("Scenario count must be positive")
    hubs = sorted(matrix.points - matrix.scores.keys())
    if len(hubs) < 2:
        raise ValueError("At least two hubs are required")
    rng = random.Random(seed)
    windows = [(300, 629), (630, 719), (720, 989), (990, 1139), (1140, 1439)]
    result = []
    for i in range(count):
        start, end = rng.sample(hubs, 2)
        lo, hi = windows[i % len(windows)]
        result.append(
            (start, end, rng.randint(lo, hi), float(rng.randint(60, 240) * 60))
        )
    return result


class Features:
    """Encode graph, dynamic state and the exact action mask in one Box observation."""

    def __init__(self, matrix: ODMatrix, coordinates: dict[str, tuple[float, float]]):
        self.matrix = matrix
        self.points = tuple(sorted(matrix.points))
        self.index = {p: i for i, p in enumerate(self.points)}
        if set(coordinates) != set(self.points):
            raise ValueError("Coordinates do not match matrix points")
        lats = [coordinates[p][0] for p in self.points]
        lons = [coordinates[p][1] for p in self.points]

        def norm(v, values):
            return (v - min(values)) / max(max(values) - min(values), 1e-9)

        self.static = np.asarray(
            [
                (
                    norm(coordinates[p][0], lats),
                    norm(coordinates[p][1], lons),
                    matrix.scores.get(p, 0.0) / 5.0,
                )
                for p in self.points
            ],
            dtype=np.float32,
        )
        n = len(self.points)
        self.adjacency = {
            slot: np.zeros((n, n), dtype=np.float32)
            for slot in ("morning", "ban_noon", "afternoon", "ban_evening", "night")
        }
        for u, v, slot in matrix.legs:
            # Rows are destinations: adjacency @ features aggregates predecessors.
            self.adjacency[slot][self.index[v], self.index[u]] = 1.0
        self.obs_size = n * 6 + n * n + n + 3 + n

    def make(self, env: ODRouteEnv) -> np.ndarray:
        obs = env.observation()
        dynamic = np.asarray(
            [[p in env.collected, p == env.end, p == env.current] for p in self.points],
            dtype=np.float32,
        )
        node_features = np.concatenate((self.static, dynamic), axis=1).ravel()
        candidate_times = np.asarray(
            [
                self.matrix.legs[env.current, p, obs["slot"]].duration_s / env.budget_s
                if (env.current, p, obs["slot"]) in self.matrix.legs
                else 0.0
                for p in self.points
            ],
            dtype=np.float32,
        )
        global_features = np.asarray(
            [
                obs["remaining_s"] / env.budget_s,
                obs["minute_of_day"] / 1440,
                env.budget_s / 14400,
            ],
            dtype=np.float32,
        )
        mask = np.asarray([obs["mask"][p] for p in self.points], dtype=np.float32)
        return np.concatenate(
            (
                node_features,
                self.adjacency[obs["slot"]].ravel(),
                candidate_times,
                global_features,
                mask,
            )
        )


class ODTrainingEnv(gym.Env):
    """Gymnasium adapter. Invalid random actions terminate with a penalty."""

    metadata = {"render_modes": []}

    def __init__(self, features: Features, heldout=(), scenario=None):
        super().__init__()
        self.features = features
        self.matrix = features.matrix
        self.hubs = sorted(self.matrix.points - self.matrix.scores.keys())
        if len(self.hubs) < 2:
            raise ValueError("At least two hubs are required")
        self.heldout = set(heldout)
        self.scenario = scenario
        self.action_space = spaces.Discrete(len(features.points))
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(features.obs_size,), dtype=np.float32
        )
        self.route_env = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.scenario is not None:
            env = ODRouteEnv(self.matrix, *self.scenario)
            if not any(env.action_mask().values()):
                raise ValueError("No valid initial action for this scenario")
            self.route_env = env
            return self.features.make(env), {"scenario": self.scenario}
        for _ in range(1000):
            start, end = self.np_random.choice(
                self.hubs, size=2, replace=False
            ).tolist()
            case = (
                start,
                end,
                int(self.np_random.integers(300, 1440)),
                float(self.np_random.integers(60, 241) * 60),
            )
            if case not in self.heldout:
                env = ODRouteEnv(self.matrix, *case)
                if any(env.action_mask().values()):
                    self.route_env = env
                    return self.features.make(env), {"scenario": case}
        raise ValueError("Unable to sample a scenario with a valid initial action")

    def step(self, action):
        if self.route_env is None or self.route_env.done:
            raise RuntimeError("Reset the environment before stepping")
        index = int(action)
        if not self.action_space.contains(index):
            raise ValueError("Action index is outside the action space")
        point = self.features.points[index]
        if not self.route_env.action_mask()[point]:
            self.route_env.done = True
            return (
                self.features.make(self.route_env),
                self.route_env.fail_penalty,
                True,
                False,
                {"success": False, "invalid_action": True},
            )
        _, reward, done, info = self.route_env.step(point)
        return self.features.make(self.route_env), reward, done, False, info


def graph_state(x, n, hidden_dim):
    """Read a batch of observations and run two directed mean GraphSAGE layers."""
    node_end = n * 6
    edge_end = node_end + n * n
    nodes = x[:, :node_end].reshape((-1, n, 6))
    adjacency = x[:, node_end:edge_end].reshape((-1, n, n))
    candidate = x[:, edge_end : edge_end + n, None]
    global_features = x[:, edge_end + n : edge_end + n + 3]
    mask = x[:, edge_end + n + 3 : edge_end + n + 3 + n] > 0.5
    embeddings = nodes
    for _ in range(2):
        neighbors = jnp.matmul(adjacency, embeddings) / jnp.maximum(
            adjacency.sum(axis=-1, keepdims=True), 1
        )
        embeddings = nn.relu(
            nn.Dense(hidden_dim)(jnp.concatenate((embeddings, neighbors), axis=-1))
        )
    current = jnp.sum(embeddings * nodes[:, :, 5:6], axis=1)
    return embeddings, current, candidate, global_features, mask


class MaskedGraphActor(Actor):
    n_nodes: int = 0
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        nodes, current, candidate, global_features, mask = graph_state(
            x, self.n_nodes, self.hidden_dim
        )
        n = self.n_nodes
        inp = jnp.concatenate(
            (
                jnp.broadcast_to(current[:, None, :], nodes.shape),
                nodes,
                candidate,
                jnp.broadcast_to(global_features[:, None, :], (x.shape[0], n, 3)),
            ),
            axis=-1,
        )
        logits = nn.Dense(1)(nn.relu(nn.Dense(self.hidden_dim)(inp))).squeeze(-1)
        # Dummy samples during SBX initialization may have an empty mask.
        # Real nonterminal routing states are guaranteed to have an action.
        mask = jnp.where(mask.any(axis=-1, keepdims=True), mask, jnp.ones_like(mask))
        return tfp.distributions.Categorical(logits=jnp.where(mask, logits, -1e9))


class GraphCritic(Critic):
    n_nodes: int = 0
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        _, current, _, global_features, _ = graph_state(
            x, self.n_nodes, self.hidden_dim
        )
        return nn.Dense(1)(
            nn.relu(
                nn.Dense(self.hidden_dim)(
                    jnp.concatenate((current, global_features), axis=-1)
                )
            )
        )


def make_model(features: Features, env: ODTrainingEnv, config: PPOConfig):
    if (
        min(config.total_steps, config.epochs, config.hidden_dim) <= 0
        or config.rollout_steps < 2
        or config.minibatch_size < 2
    ):
        raise ValueError(
            "Training settings must be positive and rollout/minibatch must be at least 2"
        )
    if (
        config.rollout_steps % config.minibatch_size
        or config.lr <= 0
        or not 0 <= config.gamma <= 1
        or not 0 < config.clip_eps < 1
    ):
        raise ValueError(
            "Invalid PPO settings; minibatch size must divide rollout steps"
        )
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=config.lr,
        n_steps=config.rollout_steps,
        batch_size=config.minibatch_size,
        n_epochs=config.epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_eps,
        vf_coef=config.vf_coef,
        ent_coef=config.ent_coef,
        seed=config.seed,
        verbose=0,
        policy_kwargs={
            "actor_class": MaskedGraphActor,
            "critic_class": GraphCritic,
            "features_extractor_kwargs": {
                "n_nodes": len(features.points),
                "hidden_dim": config.hidden_dim,
            },
        },
    )


def train(
    matrix: ODMatrix,
    matrix_path: str,
    config: PPOConfig,
    checkpoint: str,
    eval_count: int = 20,
    callback=None,
) -> list[dict]:
    heldout = scenarios(matrix, eval_count, config.seed + 1_000_000)
    features = Features(matrix, load_coordinates(matrix_path))
    env = ODTrainingEnv(features, heldout)
    model = make_model(features, env, config)
    model.matrix_sha256 = matrix_hash(matrix_path)
    model.routing_config = asdict(config)
    model.routing_eval_count = eval_count
    model.learn(total_timesteps=config.total_steps, callback=callback)
    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    return [{"steps": model.num_timesteps, "requested_steps": config.total_steps}]


def load_policy(path: str, matrix_path: str):
    model = PPO.load(path)
    if getattr(model, "matrix_sha256", None) != matrix_hash(matrix_path):
        raise ValueError("Checkpoint matrix hash does not match the supplied matrix")
    return model


def infer(
    model, features: Features, start: str, end: str, minute: int, budget_s: float
):
    env = ODRouteEnv(features.matrix, start, end, minute, budget_s)
    while not env.done and any(env.action_mask().values()):
        action, _ = model.predict(features.make(env), deterministic=True)
        env.step(features.points[int(action)])
    if env.current != end:
        return None
    return features.matrix.evaluate(env.route, minute, budget_s)


class PPOSearch(_Search):
    """Apply a trained PPO policy through the shared route solver interface."""

    def __init__(self, model, features: Features):
        super().__init__(iterations=0)
        self.model = model
        self.features = features

    def __call__(
        self, matrix: ODMatrix, start: str, end: str, start_minute: int, budget_s: float
    ):
        if matrix is not self.features.matrix:
            raise ValueError("PPO features must use the supplied matrix")
        result = infer(self.model, self.features, start, end, start_minute, budget_s)
        if result is None:
            raise ValueError("PPO did not find a feasible route")
        return result


class OnlinePPOSearch(_Search):
    """Learn on repeated simulations of each request, retaining the live policy."""

    def __init__(self, features: Features, config: PPOConfig, callback=None):
        super().__init__(iterations=0)
        self.features = features
        self.config = config
        self.callback = callback
        self.model = None

    def __call__(self, matrix, start, end, start_minute, budget_s):
        if matrix is not self.features.matrix:
            raise ValueError("PPO features must use the supplied matrix")
        env = ODTrainingEnv(self.features, scenario=(start, end, start_minute, budget_s))
        # Validate before constructing or changing a model.
        env.reset(seed=self.config.seed)
        if self.model is None:
            self.model = make_model(self.features, env, self.config)
        else:
            self.model.set_env(env)
        self.model.learn(total_timesteps=self.config.total_steps, reset_num_timesteps=False, callback=self.callback)
        result = infer(self.model, self.features, start, end, start_minute, budget_s)
        if result is None:
            raise ValueError("PPO did not find a feasible route")
        return result
