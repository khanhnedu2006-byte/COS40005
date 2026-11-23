"""Road-routing rules and the Gymnasium interface consumed by SB3.

This module contains no neural-network or PPO update code. ``RoadEnv`` owns
one trip; ``RoadTrainingEnv`` samples trips and exposes Gymnasium's reset/step
API. ``RoadFeatures`` stores the immutable graph once and builds small local
observations, so a rollout never contains a copy of the 93k-node graph.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, TypedDict, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from busroute.routing.road_graph import RoadGraph
from busroute.utils.timeslots import slot_of

FloatArray: TypeAlias = NDArray[np.float32]
IndexArray: TypeAlias = NDArray[np.signedinteger[Any]]
RoadScenario: TypeAlias = tuple[int, int, int, float]


class RoadObservation(TypedDict):
    """One local decision, before packing into SB3's float32 Box.

    Attributes:
        current: Current row position in the road-node table.
        target: Destination row position in the road-node table.
        candidates: Candidate node positions, shape (K,); padding is masked.
        edges: Edge row positions, shape (K,); -1 denotes padding.
        edge_features: Normalized attributes with shape (K, 6).
        globals: Clock, budget and visited-POI features, shape (5 + P,).
        mask: Valid outgoing action slots, shape (K,).
    """

    current: int
    target: int
    candidates: IndexArray
    edges: IndexArray
    edge_features: FloatArray
    globals: FloatArray
    mask: NDArray[np.bool_]


@dataclass
class RoadPPOConfig:
    """Configure SB3 learning and road-routing rewards.

    Time penalties are per travel minute; the simulator stores elapsed time in
    seconds. SB3 completes whole rollouts, so actual interactions can exceed
    ``total_steps``. The edge cap is a failed trip, not a Gym time truncation.

    Attributes:
        total_steps: Additional requested edge interactions for this session.
        rollout_steps: Interactions collected before each PPO update.
        minibatch_size: Samples per optimizer minibatch; divides rollout_steps.
        epochs: Optimizer passes over each rollout.
        hidden_dim: Width of each graph layer and actor/critic hidden layer.
        lr: Adam learning rate.
        gamma: Discount factor for future reward.
        gae_lambda: GAE bias/variance tradeoff.
        clip_eps: PPO policy-ratio clipping range.
        ent_coef: Weight of exploration through the entropy term.
        vf_coef: Weight of the critic's squared-error loss.
        max_grad_norm: Gradient norm limit applied by SB3.
        seed: Fresh-training seed for sampling and network initialization.
        threads: Torch CPU worker threads; four avoids excessive small-op overhead.
        max_episode_steps: Maximum traversed edges before a trip fails.
        curriculum_steps: Interactions over which trip separation grows; zero
            starts with the final sampling mixture.
        time_penalty: Reward deducted per travel minute.
        narrow_penalty: Extra deduction per minute on known roads below 5.5 m.
        arrival_bonus: Reward added on reaching the destination within budget.
        failure_penalty: Reward deducted when a trip ends unsuccessfully.
    """

    total_steps: int = 200_000
    rollout_steps: int = 256
    minibatch_size: int = 64
    epochs: int = 4
    hidden_dim: int = 32
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 0
    threads: int = 4
    max_episode_steps: int = 2048
    curriculum_steps: int = 50_000
    time_penalty: float = 0.01
    narrow_penalty: float = 0.05
    arrival_bonus: float = 10.0
    failure_penalty: float = 20.0

    def validate(self) -> None:
        """Validate PPO dimensions, coefficient ranges and reward settings.

        Raises:
            ValueError: A count, minibatch division or numeric coefficient is invalid.
        """
        counts = (
            self.total_steps,
            self.rollout_steps,
            self.minibatch_size,
            self.epochs,
            self.hidden_dim,
            self.max_episode_steps,
            self.threads,
        )
        if any(type(n) is not int or n <= 0 for n in counts):
            raise ValueError("Training counts must be positive integers")
        if self.rollout_steps < 2 or self.minibatch_size < 2:
            raise ValueError("SB3 rollout and minibatch sizes must be at least two")
        if self.rollout_steps % self.minibatch_size:
            raise ValueError("Minibatch size must divide rollout steps")
        if type(self.curriculum_steps) is not int or self.curriculum_steps < 0:
            raise ValueError("Curriculum steps must be a non-negative integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Seed must be a non-negative integer")
        if (
            not 0 <= self.gamma <= 1
            or not 0 <= self.gae_lambda <= 1
            or not 0 < self.clip_eps < 1
        ):
            raise ValueError("Invalid PPO discount or clipping settings")
        if any(not math.isfinite(v) or v <= 0 for v in (self.lr, self.max_grad_norm)):
            raise ValueError(
                "Learning rate and gradient limit must be positive and finite"
            )
        rewards = (
            self.ent_coef,
            self.vf_coef,
            self.time_penalty,
            self.narrow_penalty,
            self.arrival_bonus,
            self.failure_penalty,
        )
        if any(not math.isfinite(v) or v < 0 for v in rewards):
            raise ValueError(
                "Reward and loss coefficients must be finite and non-negative"
            )


class RoadFeatures:
    """Store full-graph attributes and build compact local observations.

    A Discrete action selects a slot among the current node's outgoing edges.
    Parallel edges remain distinct. Reverse Dijkstra only supplies a lower
    bound for masking/features; it never chooses the policy's next action.

    Attributes:
        graph: Shared road graph, including all nodes and directed edges.
        degree: Outgoing edge counts, shape (N,).
        incoming: Incoming edge counts, shape (N,).
        k: Actual maximum outgoing degree, with a minimum padding width of one.
        outgoing: Edge positions, shape (N, K); -1 denotes a padded slot.
        reverse: Sparse reverse graph weighted by minimum edge travel seconds.
        return_cache: Up to eight cached destination lower-bound vectors.
        x: Static node features, float32 array with shape (N, 6).
        global_dim: Five trip features plus one visited bit per POI.
        obs_size: Flattened Box width, 2 + 8*K + global_dim.
    """

    def __init__(self, graph: RoadGraph) -> None:
        """Precompute static features and the sparse return-bound graph.

        Args:
            graph: Loaded road graph with positive edge lengths and travel times.
        """
        self.graph = graph
        n = len(graph.nodes)
        self.degree: IndexArray = np.bincount(graph.u, minlength=n)
        self.incoming: IndexArray = np.bincount(graph.v, minlength=n)
        # NumPy's overloaded reduction hints reject valid integer arrays in ty.
        self.k = max(1, int(self.degree.max()))  # ty: ignore[invalid-argument-type]
        self.outgoing = np.full((n, self.k), -1, dtype=np.int32)
        offsets = np.zeros(n, dtype=int)
        for edge, source in enumerate(graph.u):
            self.outgoing[source, offsets[source]] = edge
            offsets[source] += 1

        # Minimum over all slots is a lower bound even if the trip crosses a
        # slot boundary. Keep the best parallel edge: CSR would sum duplicates.
        minimum = np.minimum.reduce(list(graph.times.values()))  # ty: ignore[no-matching-overload]
        best: dict[tuple[int, int], float] = {}
        for edge, (source, target) in enumerate(zip(graph.u, graph.v)):
            key = int(target), int(source)
            best[key] = min(best.get(key, math.inf), float(minimum[edge]))
        pairs = list(best)
        self.reverse = csr_matrix(
            ([best[p] for p in pairs], ([p[0] for p in pairs], [p[1] for p in pairs])),
            shape=(n, n),
        )
        self.return_cache: OrderedDict[int, NDArray[np.float64]] = OrderedDict()

        coords = graph.coords
        normalized = (coords - coords.min(axis=0)) / np.maximum(
            np.ptp(coords, axis=0), 1e-9
        )
        mask_scores = {mask: graph.mask_score(mask) for mask in set(graph.masks)}
        gains = np.array([mask_scores[mask] for mask in graph.masks])
        potential = np.zeros(n)
        np.maximum.at(potential, graph.u, gains)
        np.maximum.at(potential, graph.v, gains)
        narrow = np.bincount(
            graph.u,
            weights=np.isfinite(graph.widths) & (graph.widths < 5.5),
            minlength=n,
        )
        # Six static features: lon/lat, outgoing/incoming degree, incident POI
        # potential, and fraction of outgoing roads with a known narrow width.
        self.x = np.column_stack(
            (
                normalized,
                np.log1p(self.degree) / 3,
                np.log1p(self.incoming) / 3,
                potential / 10,
                narrow / np.maximum(self.degree, 1),
            )
        ).astype(np.float32)
        self.global_dim = 5 + len(graph.pois)
        # Flat Box layout: current, goal, K candidate nodes, K*6 edge features,
        # global state, K mask bits. A flat Box keeps local decisions compact and easy to batch.
        self.obs_size = 2 + 8 * self.k + self.global_dim

    def returns(self, target: int) -> NDArray[np.float64]:
        """Get optimistic travel seconds from every node to a destination.

        Args:
            target: Destination row position in the node table.

        Returns:
            Float64 array of shape (N,); unreachable nodes have infinite cost.
            The vector is cached, with at most eight destinations retained.

        Raises:
            InterruptedError: The graph's cancellation callback requests a stop.
        """
        if target not in self.return_cache:
            self.graph.check_cancelled()
            self.return_cache[target] = dijkstra(
                self.reverse, directed=True, indices=target
            )
            if len(self.return_cache) > 8:
                self.return_cache.popitem(last=False)
        self.return_cache.move_to_end(target)
        return self.return_cache[target]

    def valid_mask(self, env: RoadEnv) -> NDArray[np.bool_]:
        """Compute valid outgoing action slots for the current trip state.

        Excludes padded slots, self-loop edges (which cannot move closer to the
        target or pass new POIs), and edges whose immediate travel time plus
        optimistic return bound exceeds the remaining budget.

        Args:
            env: Active trip whose node, slot, target and budget are checked.

        Returns:
            Boolean mask of shape (K,).
        """
        graph = self.graph
        ids = self.outgoing[env.current]
        safe = np.maximum(ids, 0)
        times = graph.times[env.slot][safe]
        lower = self.returns(env.target)
        return (
            (ids >= 0)
            & (graph.u[safe] != graph.v[safe])
            & (times + lower[graph.v[safe]] <= env.remaining + 1e-7)
        )

    def observation(self, env: RoadEnv) -> RoadObservation:
        """Build local candidates, edge attributes, trip state and action mask.

        Args:
            env: Trip whose current node, clock, budget and visited POIs are encoded.

        Returns:
            Named arrays for one decision. K is maximum outgoing degree; edge
            features have shape (K, 6), globals (5 + P,), and mask (K,). False
            mask entries represent padding, self-loops or provably insufficient
            return time.
        """
        graph = self.graph
        ids = self.outgoing[env.current]
        valid_slot = ids >= 0
        safe = np.maximum(ids, 0)
        lower = self.returns(env.target)
        times = graph.times[env.slot][safe]
        mask = (
            valid_slot
            & (graph.u[safe] != graph.v[safe])
            & (times + lower[graph.v[safe]] <= env.remaining + 1e-7)
        )
        visited = [bool(env.collected & (1 << bit)) for bit in range(len(graph.pois))]
        gain = [
            graph.mask_score(graph.masks[e] & ~env.collected) if e >= 0 else 0
            for e in ids
        ]
        edge_features = np.column_stack(
            (
                times / env.budget_s,
                graph.lengths[safe] / 1000,
                np.isfinite(graph.widths[safe]) & (graph.widths[safe] < 5.5),
                np.isnan(graph.widths[safe]),
                np.minimum(lower[graph.v[safe]] / env.budget_s, 10),
                np.array(gain) / 10,
            )
        ).astype(np.float32)
        edge_features[~valid_slot] = 0.0
        angle = 2 * math.pi * ((env.start_minute * 60 + env.elapsed_s) % 86400) / 86400
        global_features = np.array(
            [
                env.remaining / env.budget_s,
                math.sin(angle),
                math.cos(angle),
                env.budget_s / 14400,
                min(lower[env.current] / env.budget_s, 10),
                *visited,
            ],
            dtype=np.float32,
        )
        return {
            "current": env.current,
            "target": env.target,
            "candidates": np.where(valid_slot, graph.v[safe], 0),
            "edges": ids.copy(),
            "edge_features": edge_features,
            "globals": global_features,
            "mask": mask,
        }

    def pack(self, env: RoadEnv) -> FloatArray:
        """Flatten a decision into the float32 layout decoded by the policy.

        Args:
            env: Trip to encode without copying the immutable full graph.

        Returns:
            Array of shape (obs_size,): current/goal indices, candidate indices,
            edge features, global trip state, and mask bits, in that order.
        """
        obs = self.observation(env)
        return np.concatenate(
            (
                [obs["current"], obs["target"]],
                obs["candidates"],
                obs["edge_features"].ravel(),
                obs["globals"],
                obs["mask"],
            )
        ).astype(np.float32)


class RoadEnv:
    """Simulate one trip with an actual road edge per action.

    POI bits prevent repeated rewards. The departure slot is reconsidered
    before every edge, sharing the clock used by other road solvers.

    Attributes:
        features: Shared feature builder and outgoing-action lookup.
        graph: Immutable road graph shared with the renderer.
        config: Reward coefficients and episode edge limit.
        source: Starting node row position.
        target: Destination node row position.
        start_minute: Departure minute after local midnight, in [0, 1440).
        current: Current node row position.
        budget_s: Positive trip time budget in seconds.
        elapsed_s: Accumulated edge travel seconds.
        collected: Bit mask of POIs already encountered, including the start.
        path: Traversed edge row positions, preserving parallel-edge identity.
        steps: Number of edges traversed so far.
        done: Whether arrival, budget, edge limit or a dead end ended the trip.
        success: Whether the destination was reached within the time budget.
    """

    def __init__(
        self,
        features: RoadFeatures,
        source: int,
        target: int,
        start_minute: int,
        budget_min: float,
        config: RoadPPOConfig,
    ) -> None:
        """Initialize an untraversed trip between distinct road nodes.

        Args:
            features: Shared graph features and return-time cache.
            source: Starting node row position.
            target: Distinct destination node row position.
            start_minute: Departure minute after midnight, in [0, 1440).
            budget_min: Positive finite travel-time budget in minutes.
            config: Routing reward coefficients and maximum edge count.

        Raises:
            ValueError: Endpoints, departure minute or budget are invalid.
        """
        n = len(features.graph.nodes)
        if (
            isinstance(source, bool)
            or isinstance(target, bool)
            or not isinstance(source, (int, np.integer))
            or not isinstance(target, (int, np.integer))
            or source == target
            or not 0 <= source < n
            or not 0 <= target < n
        ):
            raise ValueError("Choose distinct valid road nodes")
        if (
            isinstance(start_minute, bool)
            or not isinstance(start_minute, int)
            or not 0 <= start_minute < 1440
            or not math.isfinite(budget_min)
            or budget_min <= 0
        ):
            raise ValueError("Invalid departure or time budget")
        source, target = int(source), int(target)
        self.features, self.graph, self.config = features, features.graph, config
        self.source, self.target, self.start_minute = source, target, start_minute
        self.current, self.budget_s, self.elapsed_s = source, budget_min * 60, 0.0
        self.collected = self.graph.node_mask(source)
        self.path: list[int] = []
        self.steps = 0
        self.done, self.success = False, False

    @property
    def slot(self) -> str:
        """Get the current edge's departure-slot name.

        Returns:
            Slot determined from the trip clock, wrapping around at midnight.
        """
        return slot_of((self.start_minute + int(self.elapsed_s // 60)) % 1440)

    @property
    def remaining(self) -> float:
        """Get the unused travel-time budget.

        Returns:
            Remaining seconds, clamped to zero after budget exhaustion.
        """
        return max(0.0, self.budget_s - self.elapsed_s)

    def observe(self) -> RoadObservation:
        """Describe the current trip decision without changing its state.

        Returns:
            Candidates, attributes, global state and validity mask from RoadFeatures.
        """
        return self.features.observation(self)

    def step(self, action: int) -> tuple[float, bool]:
        """Traverse one valid outgoing edge and update reward and trip status.

        Args:
            action: Outgoing slot index, not a global edge or node identifier.

        Returns:
            Pair of shaped reward and terminal status. Reward credits newly passed
            POIs, deducts travel/narrow-road time, and adds the terminal bonus or
            penalty. Arrival, budget exhaustion, edge limit and dead ends terminate.

        Raises:
            ValueError: The trip already ended, or the action is invalid or masked.
            InterruptedError: The graph's cancellation callback requests a stop.
        """
        self.graph.check_cancelled()
        mask = self.features.valid_mask(self)
        if self.done or not 0 <= action < len(mask) or not mask[action]:
            raise ValueError("Invalid or masked road action")
        edge = int(self.features.outgoing[self.current, action])
        duration = float(self.graph.times[self.slot][edge])
        new_pois = self.graph.masks[edge] & ~self.collected
        reward = (
            self.graph.mask_score(new_pois) - self.config.time_penalty * duration / 60
        )
        if np.isfinite(self.graph.widths[edge]) and self.graph.widths[edge] < 5.5:
            reward -= self.config.narrow_penalty * duration / 60
        self.collected |= self.graph.masks[edge]
        self.path.append(edge)
        self.current = int(self.graph.v[edge])
        self.elapsed_s += duration
        self.steps += 1
        self.success = (
            self.current == self.target and self.elapsed_s <= self.budget_s + 1e-7
        )
        self.done = (
            self.success
            or self.elapsed_s > self.budget_s + 1e-7
            or self.steps >= self.config.max_episode_steps
        )
        if not self.done and not self.features.valid_mask(self).any():
            self.done = True
        if self.success:
            reward += self.config.arrival_bonus
        elif self.done:
            reward -= self.config.failure_penalty
        return float(reward), self.done


def sample_env(
    features: RoadFeatures,
    rng: np.random.Generator,
    config: RoadPPOConfig,
    steps: int,
    heldout: Sequence[Sequence[int | float]] = (),
) -> RoadEnv:
    """Sample a reachable curriculum trip without reducing the road graph.

    Args:
        features: Shared full-graph features and cached lower-bound vectors.
        rng: NumPy generator controlling scenario sampling.
        config: Curriculum horizon, rewards and trip edge limit.
        steps: Cumulative training interactions, including earlier sessions.
        heldout: Cases or endpoint pairs whose first two entries must be
            excluded as ordered training source/destination pairs.

    Returns:
        A fresh trip with at least one initially valid action. Departure is a
        local minute of day; budget is 1.5-2.5 times the optimistic bound,
        clamped to one through 240 minutes where feasible.

    Raises:
        ValueError: No feasible, non-held-out trip is found in 200 attempts.
        InterruptedError: The graph's cancellation callback requests a stop.
    """
    n = len(features.graph.nodes)
    fraction = (
        min(1.0, steps / config.curriculum_steps) if config.curriculum_steps else 1.0
    )
    excluded = {(int(case[0]), int(case[1])) for case in heldout}
    configured = sorted(set(features.graph.snapped.values()))
    for _ in range(200):
        source = target = int(rng.integers(n))
        if fraction > 0.5 and len(configured) > 1 and rng.random() < 0.25:
            source, target = map(int, rng.choice(configured, size=2, replace=False))
        elif rng.random() < fraction * 0.5:
            target = int(rng.integers(n))
        else:
            # Short directed random walks provide reachable starting examples.
            for _ in range(int(rng.integers(2, max(3, int(5 + 100 * fraction))))):
                outgoing = features.outgoing[target]
                outgoing = outgoing[outgoing >= 0]
                if not len(outgoing):
                    break
                target = int(features.graph.v[int(rng.choice(outgoing))])
        if source == target or (source, target) in excluded:
            continue
        lower = float(features.returns(target)[source])
        if not math.isfinite(lower) or lower <= 0:
            continue
        budget = min(14400.0, max(60.0, lower * float(rng.uniform(1.5, 2.5)))) / 60
        if budget * 60 < lower * 1.05:
            continue
        env = RoadEnv(features, source, target, int(rng.integers(1440)), budget, config)
        if features.valid_mask(env).any():
            return env
    raise ValueError("Could not sample a feasible training scenario")


class RoadTrainingEnv(gym.Env[FloatArray, np.int64]):
    """Expose road trips through Gymnasium's reset/step interface for SB3.

    All routing endings use terminated=True. The edge cap means failure, so
    using truncated=True would incorrectly bootstrap reward beyond that trip.
    SB3 independently bootstraps nonterminal rollout-buffer cuts.

    Attributes:
        features: Shared full-graph feature builder.
        config: Reward and sampling settings.
        heldout: Ordered endpoint pairs excluded from random training trips.
        scenario: Fixed (source, target, departure_minute, budget_min) case,
            or None for curriculum sampling.
        initial_steps: Interaction offset restored when resetting with a seed.
        training_steps: Cumulative interactions used by the curriculum.
        pending_rng_state: Saved sampler state consumed once on the next reset.
        route_env: Active trip, or None until the first reset.
        action_space: Discrete outgoing-slot indices.
        observation_space: Float32 Box of shape (features.obs_size,).
    """

    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012 -- Gymnasium declares an instance attribute.

    def __init__(
        self,
        features: RoadFeatures,
        config: RoadPPOConfig,
        heldout: Sequence[Sequence[int | float]] = (),
        *,
        scenario: RoadScenario | None = None,
        steps: int = 0,
        rng_state: dict[str, Any] | None = None,
    ) -> None:
        """Create an adapter without starting its first trip.

        Args:
            features: Shared full-graph attributes and action lookup.
            config: PPO routing and curriculum settings.
            heldout: Ordered training endpoint pairs to exclude.
            scenario: Optional fixed node-position case for evaluation/checks.
            steps: Cumulative interaction offset for resumed curriculum sampling.
            rng_state: Optional NumPy bit-generator state restored on first reset.
        """
        super().__init__()
        self.features, self.config = features, config
        self.heldout, self.scenario = heldout, scenario
        self.initial_steps = self.training_steps = steps
        self.pending_rng_state = rng_state
        self.route_env: RoadEnv | None = None
        self.action_space: spaces.Space[np.int64] = spaces.Discrete(features.k)
        self.observation_space: spaces.Space[FloatArray] = cast(
            spaces.Space[FloatArray],
            spaces.Box(
                -np.inf,
                np.inf,
                shape=(features.obs_size,),
                dtype=np.float32,
            ),
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        """Start a fixed or sampled trip, honoring Gymnasium seeding.

        Args:
            seed: Optional seed; also resets the curriculum to initial_steps.
            options: Gymnasium compatibility argument; currently unused.

        Returns:
            Initial float32 observation and an empty Gymnasium info dictionary.

        Raises:
            ValueError: The fixed case has no valid action, or sampling fails.
            InterruptedError: The graph's cancellation callback requests a stop.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.training_steps = self.initial_steps
        if self.pending_rng_state is not None:
            self.np_random.bit_generator.state = self.pending_rng_state
            self.pending_rng_state = None
        if self.scenario is None:
            self.route_env = sample_env(
                self.features,
                self.np_random,
                self.config,
                self.training_steps,
                self.heldout,
            )
        else:
            self.route_env = RoadEnv(self.features, *self.scenario, self.config)
            if not self.features.valid_mask(self.route_env).any():
                raise ValueError("No valid initial action for this scenario")
        return self.features.pack(self.route_env), {}

    def step(
        self,
        action: int | np.int64,
    ) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        """Adapt one road action to Gymnasium's five-element step result.

        Args:
            action: Outgoing slot, normally a NumPy integer supplied by SB3.

        Returns:
            Observation, shaped reward, terminated flag, False for truncation, and
            an info dictionary whose success flag distinguishes arrival/failure.
            Invalid random checker actions terminate with the failure penalty.

        Raises:
            RuntimeError: No active trip exists, or the current trip already ended.
            InterruptedError: The graph's cancellation callback requests a stop.
        """
        if self.route_env is None or self.route_env.done:
            raise RuntimeError("Reset the environment before stepping")
        self.training_steps += 1
        index = int(action)
        if (
            not self.action_space.contains(index)
            or not self.features.valid_mask(self.route_env)[index]
        ):
            # Generic Gym environment checkers sample unmasked random actions.
            # A real masked policy cannot enter this branch.
            self.route_env.done = True
            reward, done = -self.config.failure_penalty, True
        else:
            reward, done = self.route_env.step(index)
        info = {"success": self.route_env.success}
        return self.features.pack(self.route_env), reward, bool(done), False, info
