"""Budget-safe waypoint decisions over the shared directed road planner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from busroute.routing.road_graph import (
    NoRoadRouteError,
    RoadGraph,
    RoadLeg,
    RoadProblem,
    WaypointConnections,
)

Scenario = tuple[str, str, int, float]
BENCHMARK_CASES: tuple[Scenario, ...] = (
    ("H01", "H02", 480, 240),
    ("H01", "H02", 625, 120),
    ("H01", "H05", 480, 70),
    ("H01", "H05", 1435, 70),
    ("H02", "H01", 480, 120),
    ("H03", "H04", 985, 70),
)
TRAIN_DEPARTURES = (480, 675, 870, 1065, 1230)
TRAIN_BUDGETS = (60, 70, 120, 180, 240)
REWARD_SCALE = 10.0
TIME_TOLERANCE = 1e-7
ACTION_FEATURES = 7


class WaypointFeatures:
    """Graph and connection data shared across training, inference and tests."""

    def __init__(self, graph: RoadGraph) -> None:
        self.graph = graph
        self.connections = WaypointConnections(graph)
        self.score = lru_cache(maxsize=32768)(graph.mask_score)
        self.action_ids = (
            *graph.pois,
            None,
        )  # None is Finish at the requested endpoint.
        self.poi_bits = {poi: 1 << i for i, poi in enumerate(graph.pois)}
        self.finish = len(self.action_ids) - 1
        self.global_features = 8 + self.finish
        self.obs_size = self.global_features + len(self.action_ids) * (
            ACTION_FEATURES + 1
        )
        coords = np.array([(p["lon"], p["lat"]) for p in graph.points.values()])
        normalized = (coords - coords.min(axis=0)) / np.maximum(
            np.ptp(coords, axis=0), 1e-9
        )
        self.coords = dict(zip(graph.points, normalized))

    def problem(self, scenario: Scenario) -> RoadProblem:
        start, end, minute, budget = scenario
        return RoadProblem(
            self.graph,
            start,
            end,
            budget,
            start_minute=minute,
            waypoint_connections=self.connections,
        )

    def training_cases(self) -> tuple[Scenario, ...]:
        hubs = sorted(
            p for p in self.graph.snapped if self.graph.points[p]["kind"] == "hub"
        )
        heldout = {(start, end) for start, end, _, _ in BENCHMARK_CASES}
        cases = []
        for start, end in permutations(hubs, 2):
            if (start, end) in heldout:
                continue
            for minute in TRAIN_DEPARTURES:
                for budget in TRAIN_BUDGETS:
                    scenario = (start, end, minute, budget)
                    try:
                        self.problem(scenario)
                    except NoRoadRouteError:
                        continue
                    cases.append(scenario)
        if not cases:
            raise ValueError(
                "No feasible hub scenarios are available for waypoint training"
            )
        return tuple(cases)


@dataclass(frozen=True)
class TimedLeg:
    leg: RoadLeg
    arrival: float


@dataclass(frozen=True)
class Move:
    point: str
    outbound: TimedLeg
    continuation: TimedLeg | None


class WaypointTrip:
    """Commit a leg only when its exact continuation is already reserved."""

    def __init__(self, features: WaypointFeatures, scenario: Scenario) -> None:
        self.features = features
        self.problem = problem = features.problem(scenario)
        self.current = problem.start
        self.elapsed = 0.0
        self.route = [self.current]
        self.edges: list[int] = []
        self.selected: set[str] = set()
        self.collected = problem.graph.node_mask(problem.graph.snapped[self.current])
        slot = (problem.direct.connection_slots or problem.direct.slots)[0]
        self.reserve = TimedLeg(
            problem.leg(problem.start, problem.end, slot), problem.direct.duration_s
        )
        self.done = False
        self.moves: dict[int, Move] = {}

    def _detour(self, point: str) -> Move | None:
        problem = self.problem
        primary = problem.timed_connection(self.current, point, self.elapsed)
        if primary is None or primary[1] > problem.budget_s + TIME_TOLERANCE:
            return None
        outbound_options = [primary]
        original = problem.timed_connection(
            self.current,
            point,
            self.elapsed,
            compare_slots=False,
        )
        if original is not None and original[2] != primary[2]:
            outbound_options.append(original)
        for leg, arrival, _ in outbound_options:
            if arrival > problem.budget_s + TIME_TOLERANCE:
                continue
            result = problem.timed_connection(point, problem.end, arrival)
            if result is not None and result[1] <= problem.budget_s + TIME_TOLERANCE:
                tail, end_time, _ = result
                return Move(point, TimedLeg(leg, arrival), TimedLeg(tail, end_time))
        return None

    def observe(self) -> np.ndarray:
        features = self.features
        obs = np.zeros(features.obs_size, dtype=np.float32)
        if self.done:
            return obs
        problem = self.problem
        problem.graph.check_cancelled()
        finish = features.finish
        self.moves = {finish: Move(problem.end, self.reserve, None)}
        if problem.graph.snapped[self.current] != problem.target:
            for action, point in enumerate(features.action_ids[:-1]):
                if (
                    point not in problem.candidates
                    or point in self.selected
                    or self.collected & features.poi_bits[point]
                ):
                    continue
                move = self._detour(point)
                if move is not None:
                    self.moves[action] = move
        budget = problem.budget_s
        angle = (
            2 * math.pi * ((problem.start_minute * 60 + self.elapsed) % 86400) / 86400
        )
        obs[: features.global_features] = (
            *features.coords[self.current],
            *features.coords[problem.end],
            math.sin(angle),
            math.cos(angle),
            (budget - self.elapsed) / budget,
            budget / 14400,
            *(bool(self.collected & bit) for bit in features.poi_bits.values()),
        )
        rows = obs[features.global_features : -(finish + 1)].reshape(
            -1, ACTION_FEATURES
        )
        mask = obs[-(finish + 1) :]
        for action, move in self.moves.items():
            outbound, tail = move.outbound, move.continuation
            new = outbound.leg.collected & ~self.collected
            rows[action] = (
                *features.coords[move.point],
                (outbound.arrival - self.elapsed) / budget,
                (budget - (tail or outbound).arrival) / budget,
                features.score(new) / REWARD_SCALE,
                features.score(tail.leg.collected & ~(self.collected | new))
                / REWARD_SCALE
                if tail
                else 0.0,
                action == finish,
            )
            mask[action] = 1
        return obs

    def step(self, action: int) -> float:
        self.problem.graph.check_cancelled()
        if self.done or action not in self.moves:
            raise ValueError("Invalid or masked waypoint action")
        move = self.moves[action]
        reward = self.features.score(move.outbound.leg.collected & ~self.collected)
        self.collected |= move.outbound.leg.collected
        self.edges.extend(move.outbound.leg.edges)
        self.current = move.point
        self.route.append(self.current)
        self.elapsed = move.outbound.arrival
        self.done = move.continuation is None
        if move.continuation is not None:
            self.reserve = move.continuation
            self.selected.add(move.point)
        self.moves.clear()
        return reward / REWARD_SCALE

    def snapshot(self):
        if not self.done:
            raise ValueError("Waypoint trip has not reached its destination")
        problem = self.problem
        return problem.graph.render_path(
            self.route, self.edges, problem.start_minute, problem.budget_s
        )


class WaypointEnv(gym.Env[np.ndarray, np.int64]):
    """Sample verified scenarios and expose the waypoint policy to SB3."""

    metadata: dict[str, object] = {"render_modes": []}  # noqa: RUF012 -- Gymnasium declares an instance attribute.

    def __init__(
        self, features: WaypointFeatures, scenarios: tuple[Scenario, ...]
    ) -> None:
        super().__init__()
        if not scenarios:
            raise ValueError("WaypointEnv requires at least one scenario")
        self.features = features
        self.scenarios = scenarios
        self.trip: WaypointTrip | None = None
        self.action_space = spaces.Discrete(len(features.action_ids))
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(features.obs_size,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scenario = self.scenarios[int(self.np_random.integers(len(self.scenarios)))]
        self.trip = WaypointTrip(self.features, scenario)
        return self.trip.observe(), {}

    def step(self, action):
        if self.trip is None or self.trip.done:
            raise RuntimeError("Reset the environment before stepping")
        index = int(action)
        if not self.action_space.contains(index) or index not in self.trip.moves:
            self.trip.done = True
            self.trip.moves.clear()
            return self.trip.observe(), -1.0, True, False, {"invalid_action": True}
        reward = self.trip.step(index)
        return self.trip.observe(), reward, self.trip.done, False, {}
