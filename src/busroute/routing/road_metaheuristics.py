"""ALNS, VNS, ACO, and Dijkstra solvers on time-budgeted POI connections.

All waypoint-level solvers inherit from ``PoiBaseSolver(BaseRouteSolver)`` and share
``PoiContext``, ``RouteRequest``, ``RoadDatasetSpec``, and ``RouteOutput``.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import random
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, ClassVar

from busroute.routing.base import (
    BaseRouteSolver,
    RoadDatasetSpec,
    RouteOutput,
    RouteRequest,
    _fork_worker_dispatch,
)
from busroute.routing.road_graph import RoadGraph, RoadProblem, RoadResult
from busroute.utils.timeslots import slot_of

# Hằng số phần thưởng mặc định của WaypointBusEnv dùng cho ACO ở chế độ POI.
WAYPOINT_PPO_TIME_PENALTY = 0.005
WAYPOINT_PPO_NARROW_PENALTY = 0.5
WAYPOINT_PPO_SUCCESS_BONUS = 1.0


@dataclass(frozen=True)
class PoiContext:
    """Loaded ``RoadGraph`` and ``RoadProblem`` shared across waypoint-level solvers."""

    road_graph: RoadGraph
    problem: RoadProblem


class PoiBaseSolver(BaseRouteSolver[PoiContext, RoadResult]):
    """Base solver for POI-level routing over precomputed ``RoadProblem`` connections."""

    mode: ClassVar[str] = "poi"
    all_pois: ClassVar[bool] = True
    _fork_state: ClassVar[dict[str, Any]] = {}

    def parse_data(self, data: RoadDatasetSpec, request: RouteRequest) -> PoiContext:
        """Load ``RoadGraph`` and construct a ``RoadProblem`` for ``request``."""
        request.check_cancelled()
        road_graph = RoadGraph.from_files(
            data.nodes_path,
            data.edges_path,
            data.widths_path,
            list(data.points),
            request.cancelled,
            reduced_dir=data.reduced_dir,
            use_reduced=data.use_reduced,
        )
        problem = RoadProblem(
            road_graph,
            request.start,
            request.end,
            request.budget_min,
            start_minute=request.start_minute,
            all_pois=self.all_pois,
            workers=request.workers,
        )
        return PoiContext(road_graph=road_graph, problem=problem)

    def format_output(
        self, context: PoiContext, result: RoadResult, request: RouteRequest
    ) -> RouteOutput:
        """Render the chosen ``RoadResult`` into a standardized ``RouteOutput``."""
        snapshot = context.road_graph.snapshot(context.problem, result)
        return RouteOutput.from_snapshot(snapshot)

    @staticmethod
    def _insertions(problem: RoadProblem, current: RoadResult):
        for poi in problem.candidates:
            if poi in current.route:
                continue
            for index in range(1, len(current.route)):
                route = list(current.route)
                route.insert(index, poi)
                candidate = problem.evaluate(route)
                if candidate is not None and candidate.quality > current.quality:
                    extra = max(candidate.duration_s - current.duration_s, 1.0)
                    yield (candidate.score - current.score) / extra, candidate, index

    @classmethod
    def _repair(
        cls,
        problem: RoadProblem,
        initial: RoadResult,
        rng: random.Random,
        randomized: bool = False,
    ) -> RoadResult:
        current = initial
        while True:
            options = list(cls._insertions(problem, current))
            if not options:
                return current
            options.sort(key=lambda item: (item[0], item[1].quality), reverse=True)
            current = rng.choice(options[:3])[1] if randomized else options[0][1]

    @staticmethod
    def _remove(
        problem: RoadProblem,
        current: RoadResult,
        rng: random.Random,
        count: int,
        worst: bool = False,
    ) -> RoadResult:
        for _ in range(min(count, len(current.route) - 2)):
            options = [
                candidate
                for index in range(1, len(current.route) - 1)
                if (
                    candidate := problem.evaluate(
                        current.route[:index] + current.route[index + 1 :]
                    )
                )
                is not None
            ]
            if not options:
                break
            current = (
                max(options, key=lambda result: result.quality)
                if worst
                else rng.choice(options)
            )
        return current


class PoiDijkstraSolver(PoiBaseSolver):
    """Direct shortest-time baseline on ``RoadProblem``."""

    name: ClassVar[str] = "dijkstra"
    all_pois: ClassVar[bool] = False

    def solve(self, context: PoiContext, request: RouteRequest) -> RoadResult:
        request.check_cancelled()
        return context.problem.direct


class PoiALNSSolver(PoiBaseSolver):
    """Adaptive Large Neighbourhood Search over POI waypoints."""

    name: ClassVar[str] = "alns"

    @classmethod
    def _fork_worker(
        cls, task: tuple[int, Any, list[float], list[float]]
    ) -> tuple[int, int, Any]:
        trial_seed, current, destroy_weights, repair_weights = task
        problem = cls._fork_state["problem"]
        rng = random.Random(trial_seed)
        destroy = rng.choices(range(2), weights=destroy_weights)[0]
        repair = rng.choices(range(2), weights=repair_weights)[0]
        count = rng.randint(1, max(1, min(4, len(current.route) - 2)))
        partial = cls._remove(problem, current, rng, count, bool(destroy))
        candidate = cls._repair(problem, partial, rng, bool(repair))
        return destroy, repair, candidate

    def solve(self, context: PoiContext, request: RouteRequest) -> RoadResult:
        problem = context.problem
        rng = random.Random(request.seed)
        current = best = self._repair(problem, problem.direct, rng)
        destroy_weights, repair_weights = [1.0, 1.0], [1.0, 1.0]

        if self._can_fork(request.workers, request.iterations):
            problem.trees(slot_of(problem.start_minute))
            ctx = mp.get_context("fork")
            self._fork_state["problem"] = problem
            try:
                with ctx.Pool(processes=min(request.workers, request.iterations)) as pool:
                    done = 0
                    while done < request.iterations:
                        problem.graph.check_cancelled()
                        batch = min(request.workers, request.iterations - done)
                        tasks = [
                            (
                                type(self),
                                (
                                    request.seed * 1000003 + done + offset,
                                    current,
                                    list(destroy_weights),
                                    list(repair_weights),
                                ),
                            )
                            for offset in range(batch)
                        ]
                        outcomes = pool.map(_fork_worker_dispatch, tasks)
                        best_cand = None
                        for destroy, repair, candidate in outcomes:
                            improved_best = candidate.quality > best.quality
                            improved_current = candidate.quality > current.quality
                            if improved_best:
                                best = candidate
                            if best_cand is None or candidate.quality > best_cand.quality:
                                best_cand = candidate
                            reward = 5.0 if improved_best else 2.0 if improved_current else 0.5
                            destroy_weights[destroy] = (
                                0.8 * destroy_weights[destroy] + 0.2 * reward
                            )
                            repair_weights[repair] = (
                                0.8 * repair_weights[repair] + 0.2 * reward
                            )
                        if best_cand is not None:
                            temperature = max(
                                0.01, 1.0 - done / max(request.iterations, 1)
                            )
                            delta = best_cand.score - current.score
                            if best_cand.quality > current.quality or (
                                delta < 0 and rng.random() < math.exp(delta / temperature)
                            ):
                                current = best_cand
                        done += batch
            finally:
                self._fork_state.clear()
            return best

        for iteration in range(request.iterations):
            problem.graph.check_cancelled()
            destroy = rng.choices(range(2), weights=destroy_weights)[0]
            repair = rng.choices(range(2), weights=repair_weights)[0]
            count = rng.randint(1, max(1, min(4, len(current.route) - 2)))
            partial = self._remove(problem, current, rng, count, bool(destroy))
            candidate = self._repair(problem, partial, rng, bool(repair))
            improved_best = candidate.quality > best.quality
            improved_current = candidate.quality > current.quality
            temperature = max(0.01, 1.0 - iteration / max(request.iterations, 1))
            delta = candidate.score - current.score
            if improved_current or (
                delta < 0 and rng.random() < math.exp(delta / temperature)
            ):
                current = candidate
            if improved_best:
                best = candidate
            reward = 5.0 if improved_best else 2.0 if improved_current else 0.5
            destroy_weights[destroy] = 0.8 * destroy_weights[destroy] + 0.2 * reward
            repair_weights[repair] = 0.8 * repair_weights[repair] + 0.2 * reward
        return best


class PoiVNSSolver(PoiBaseSolver):
    """Variable Neighbourhood Search over POI waypoints."""

    name: ClassVar[str] = "vns"

    @staticmethod
    def _local_search(problem: RoadProblem, current: RoadResult) -> RoadResult:
        best = current
        unused = [p for p in problem.candidates if p not in current.route]
        for index in range(1, len(current.route) - 1):
            for poi in unused:
                route = list(current.route)
                route[index] = poi
                candidate = problem.evaluate(route)
                if candidate is not None and candidate.quality > best.quality:
                    best = candidate
            route = list(current.route)
            point = route.pop(index)
            for target in range(1, len(route)):
                trial = route[:target] + [point] + route[target:]
                candidate = problem.evaluate(trial)
                if candidate is not None and candidate.quality > best.quality:
                    best = candidate
        return best

    @classmethod
    def _fork_worker(cls, task: tuple[int, Any, int]) -> Any:
        trial_seed, best, k = task
        problem = cls._fork_state["problem"]
        rng = random.Random(trial_seed)
        partial = cls._remove(problem, best, rng, k)
        return cls._local_search(problem, cls._repair(problem, partial, rng, True))

    def solve(self, context: PoiContext, request: RouteRequest) -> RoadResult:
        problem = context.problem
        rng = random.Random(request.seed)
        best = self._repair(problem, problem.direct, rng)
        neighbourhood = 1

        if self._can_fork(request.workers, request.iterations):
            problem.trees(slot_of(problem.start_minute))
            ctx = mp.get_context("fork")
            self._fork_state["problem"] = problem
            try:
                with ctx.Pool(processes=min(request.workers, request.iterations)) as pool:
                    done = 0
                    while done < request.iterations:
                        problem.graph.check_cancelled()
                        batch = min(request.workers, request.iterations - done)
                        tasks = [
                            (
                                type(self),
                                (
                                    request.seed * 1000003 + done + offset,
                                    best,
                                    ((neighbourhood - 1 + offset) % 4) + 1,
                                ),
                            )
                            for offset in range(batch)
                        ]
                        candidates = pool.map(_fork_worker_dispatch, tasks)
                        winner = max(candidates, key=lambda res: res.quality)
                        if winner.quality > best.quality:
                            best, neighbourhood = winner, 1
                        else:
                            neighbourhood = ((neighbourhood - 1 + batch) % 4) + 1
                        done += batch
            finally:
                self._fork_state.clear()
            return best

        for _ in range(request.iterations):
            problem.graph.check_cancelled()
            partial = self._remove(problem, best, rng, neighbourhood)
            candidate = self._local_search(problem, self._repair(problem, partial, rng, True))
            if candidate.quality > best.quality:
                best, neighbourhood = candidate, 1
            else:
                neighbourhood = neighbourhood % 4 + 1
        return best


class PoiACOSolver(PoiBaseSolver):
    """Ant Colony Optimization maximizing ``WaypointBusEnv`` episode returns."""

    name: ClassVar[str] = "aco"

    @staticmethod
    def _waypoint_ppo_return(result: Any) -> float:
        """Tính tổng phần thưởng episode của WaypointBusEnv cho một kết quả POI."""
        narrow_m = max(0.0, float(result.weighted_m) - float(result.distance_m))
        return (
            float(result.score)
            - WAYPOINT_PPO_TIME_PENALTY * (float(result.duration_s) / 60.0)
            - WAYPOINT_PPO_NARROW_PENALTY * (narrow_m / 1000.0)
            + WAYPOINT_PPO_SUCCESS_BONUS
        )

    @classmethod
    def _result_key(cls, result: Any) -> tuple[float, Any]:
        """Khóa xếp hạng của ACO theo phần thưởng PPO (dùng quality khi hòa điểm)."""
        return (round(cls._waypoint_ppo_return(result), 9), result.quality)

    @classmethod
    def _aco_insertions(cls, problem: RoadProblem, current: RoadResult):
        """Sinh các phép chèn POI hợp lệ làm tăng phần thưởng WaypointBusEnv cho ACO."""
        current_return = cls._waypoint_ppo_return(current)
        for poi in problem.candidates:
            if poi in current.route:
                continue
            for index in range(1, len(current.route)):
                route = list(current.route)
                route.insert(index, poi)
                candidate = problem.evaluate(route)
                if candidate is None:
                    continue
                cand_return = cls._waypoint_ppo_return(candidate)
                if cand_return > current_return:
                    extra = max(candidate.duration_s - current.duration_s, 1.0)
                    yield (cand_return - current_return) / extra, candidate, index

    @classmethod
    def _aco_repair(cls, problem: RoadProblem, initial: RoadResult) -> RoadResult:
        """Chèn tham lam các POI theo mức tăng phần thưởng PPO mỗi giây cho hạt giống ACO."""
        current = initial
        while options := list(cls._aco_insertions(problem, current)):
            options.sort(key=lambda item: (item[0], cls._result_key(item[1])), reverse=True)
            current = options[0][1]
        return current

    @classmethod
    def _fork_worker(cls, task: tuple[int, dict[tuple[str, str], float]]) -> Any:
        ant_seed, pheromone = task
        problem = cls._fork_state["problem"]
        rng = random.Random(ant_seed)
        current = problem.direct
        while options := list(cls._aco_insertions(problem, current)):
            weights = []
            for gain_per_second, candidate, index in options:
                before, poi, after = candidate.route[index - 1 : index + 2]
                trail_in = pheromone.get((before, poi), 1.0)
                trail_out = pheromone.get((poi, after), 1.0)
                weights.append(trail_in * trail_out * (1.0 + gain_per_second))
            _, current, _ = rng.choices(options, weights=weights, k=1)[0]
        return current

    def solve(self, context: PoiContext, request: RouteRequest) -> RoadResult:
        problem = context.problem
        rng = random.Random(request.seed)
        best = self._aco_repair(problem, problem.direct)
        best_key = self._result_key(best)

        if self._can_fork(request.workers, request.iterations):
            problem.trees(slot_of(problem.start_minute))
            pheromone: dict[tuple[str, str], float] = {}
            ctx = mp.get_context("fork")
            self._fork_state["problem"] = problem
            try:
                with ctx.Pool(processes=min(request.workers, request.iterations)) as pool:
                    done = 0
                    while done < request.iterations:
                        problem.graph.check_cancelled()
                        batch = min(request.workers, request.iterations - done)
                        tasks = [
                            (
                                type(self),
                                (request.seed * 1000003 + done + offset, dict(pheromone)),
                            )
                            for offset in range(batch)
                        ]
                        candidates = pool.map(_fork_worker_dispatch, tasks)
                        for current in candidates:
                            cur_key = self._result_key(current)
                            if cur_key > best_key:
                                best, best_key = current, cur_key
                        for connection, trail in pheromone.items():
                            pheromone[connection] = max(0.1, trail * (0.9**batch))
                        for connection in pairwise(best.route):
                            pheromone[connection] = (
                                pheromone.get(connection, 1.0) + float(batch)
                            )
                        done += batch
            finally:
                self._fork_state.clear()
            return best

        pheromone = {}
        for _ in range(request.iterations):
            problem.graph.check_cancelled()
            current = problem.direct
            while options := list(self._aco_insertions(problem, current)):
                weights = []
                for gain_per_second, candidate, index in options:
                    before, poi, after = candidate.route[index - 1 : index + 2]
                    trail_in = pheromone.get((before, poi), 1.0)
                    trail_out = pheromone.get((poi, after), 1.0)
                    weights.append(trail_in * trail_out * (1.0 + gain_per_second))
                _, current, _ = rng.choices(options, weights=weights, k=1)[0]
            cur_key = self._result_key(current)
            if cur_key > best_key:
                best, best_key = current, cur_key
            for connection, trail in pheromone.items():
                pheromone[connection] = max(0.1, trail * 0.9)
            for connection in pairwise(best.route):
                pheromone[connection] = pheromone.get(connection, 1.0) + 1.0
        return best


POI_SOLVERS: dict[str, type[PoiBaseSolver]] = {
    "dijkstra": PoiDijkstraSolver,
    "alns": PoiALNSSolver,
    "vns": PoiVNSSolver,
    "aco": PoiACOSolver,
}


# Các hàm bọc tương thích ngược cho caller hiện có và bộ kiểm thử.


def _waypoint_ppo_return(result: Any) -> float:
    return PoiACOSolver._waypoint_ppo_return(result)


def search(
    problem: RoadProblem,
    algorithm: str,
    *,
    iterations: int = 100,
    seed: int = 0,
    workers: int = 1,
) -> RoadResult:
    """Return the best feasible route. Connections reuse cached trees for each departure slot."""
    if algorithm not in ("alns", "vns", "aco"):
        raise ValueError("Choose alns, vns or aco")
    request = RouteRequest(
        start=problem.start,
        end=problem.end,
        budget_min=problem.budget_s / 60.0,
        start_minute=problem.start_minute,
        iterations=iterations,
        seed=seed,
        workers=workers,
        cancelled=problem.graph.cancelled,
    )
    solver = POI_SOLVERS[algorithm]()
    return solver.solve(PoiContext(road_graph=problem.graph, problem=problem), request)
