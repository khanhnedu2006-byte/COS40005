"""Basic ALNS and VNS search for the time-dependent orienteering problem."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Protocol

from busroute.routing.orienteering import ODMatrix, RouteResult


class Algorithm(Protocol):
    """A callable route solver for a directed, time-dependent OD matrix.

    The request contains distinct start/end IDs, departure minutes after
    midnight, and a time budget in seconds. Implementations return the best
    feasible route found or raise ValueError if an initial route is unavailable.
    """

    def __call__(
        self, matrix: ODMatrix, start: str, end: str, start_minute: int, budget_s: float
    ) -> RouteResult: ...


class _Search:
    """Shared search operations and configuration.

    Args:
        iterations: Number of search iterations after initial construction.
        seed: Random seed; each call starts a fresh reproducible generator.
        progress: Optional callback receiving JSON-compatible progress records.
        trace: Also emit selected operator actions for live visualization.
    """

    def __init__(self, *, iterations: int = 100, seed: int = 0,
                 progress: Callable | None = None, trace: bool = False) -> None:
        self._check(iterations)
        self.iterations = iterations
        self.seed = seed
        self.progress = progress
        self.trace = trace
        self._iteration = 0
        self._sequence = 0

    def _log(self, event: str, **fields) -> None:
        if event == "search_start":
            self._sequence = 0
            self._iteration = 0
        if self.progress is not None:
            self._sequence += 1
            self.progress({"version": 1, "sequence": self._sequence,
                           "event": event, **fields})

    def _log_best(self, event: str, iteration: int, best: RouteResult, **fields) -> None:
        self._log(event, iteration=iteration, iterations=self.iterations,
                  best_score=best.score, best_duration_min=best.duration_s / 60,
                  best=best.to_dict(), **fields)

    def _move(self, action: str, result: RouteResult, **fields) -> None:
        """Emit selected operator actions, never individual trial evaluations."""
        if self.trace:
            self._log("operator", iteration=self._iteration, iterations=self.iterations,
                      action=action, candidate=result.to_dict(), **fields)

    @staticmethod
    def _quality(result: RouteResult) -> tuple[float, float]:
        """Maximise score first; prefer a shorter route only when scores tie."""
        return result.score, -result.duration_s

    @staticmethod
    def _initial(
        matrix: ODMatrix, start: str, end: str, minute: int, budget: float
    ) -> RouteResult:
        """Find a feasible direct route or a route via one candidate POI."""
        if start == end or start not in matrix.points or end not in matrix.points:
            raise ValueError("Start and end must be distinct known points")
        direct = matrix.evaluate([start, end], minute, budget)
        if direct is not None:
            return direct
        # A direct leg can be unavailable even when a one-stop route is possible.
        candidates = (
            matrix.evaluate([start, p, end], minute, budget)
            for p in sorted(matrix.scores)
            if p not in (start, end)
        )
        feasible = [result for result in candidates if result is not None]
        if not feasible:
            raise ValueError(
                "No feasible direct or one-stop initial route found; a longer feasible route may exist"
            )
        return max(feasible, key=_Search._quality)

    def _repair(
        self,
        matrix: ODMatrix,
        initial: RouteResult,
        minute: int,
        budget: float,
        rng: random.Random,
        randomized: bool = False,
    ) -> RouteResult:
        """Repeatedly insert an unselected stop when it improves route quality."""
        current = initial
        while True:
            options: list[tuple[float, RouteResult]] = []
            for poi in sorted(matrix.scores.keys() - set(current.route)):
                for index in range(1, len(current.route)):
                    route = list(current.route)
                    route.insert(index, poi)
                    candidate = matrix.evaluate(route, minute, budget)
                    if candidate is not None and _Search._quality(
                        candidate
                    ) > _Search._quality(current):
                        gain = candidate.score - current.score
                        extra = max(candidate.duration_s - current.duration_s, 1.0)
                        options.append((gain / extra, candidate))
            if not options:
                return current
            options.sort(
                key=lambda item: (item[0], *_Search._quality(item[1])), reverse=True
            )
            current = (
                rng.choice(options[: min(3, len(options))])[1]
                if randomized
                else options[0][1]
            )
            self._move("insert", current)

    def _remove(
        self,
        matrix: ODMatrix,
        current: RouteResult,
        minute: int,
        budget: float,
        rng: random.Random,
        count: int,
        worst: bool,
    ) -> RouteResult:
        """Remove stops without ever leaving the feasible solution space."""
        for _ in range(min(count, len(current.route) - 2)):
            options = []
            for index in range(1, len(current.route) - 1):
                route = list(current.route)
                del route[index]
                candidate = matrix.evaluate(route, minute, budget)
                if candidate is not None:
                    options.append(candidate)
            if not options:
                break
            current = (
                max(options, key=_Search._quality) if worst else rng.choice(options)
            )
            self._move("remove", current)
        return current

    @staticmethod
    def _check(iterations: int) -> None:
        """Reject a negative iteration count."""
        if iterations < 0:
            raise ValueError("iterations must be non-negative")


class ALNS(_Search):
    """Search for a high-scoring route with adaptive large neighbourhood search.

    Starting with a feasible route, ALNS repeatedly removes stops and inserts
    promising POIs. It adjusts the selection weights of its removal and repair
    methods based on their results. Simulated annealing sometimes accepts a
    lower-scoring route to explore alternatives, while preserving the best
    route found. Score is maximised first; travel time breaks score ties.

    Args:
        iterations: Number of destroy-and-repair iterations. Defaults to 100.
        seed: Seed for reproducible random choices. Defaults to 0.

    Raises:
        ValueError: If iterations is negative.
    """

    def __call__(
        self, matrix: ODMatrix, start: str, end: str, start_minute: int, budget_s: float
    ) -> RouteResult:
        """Return the best feasible route found for the supplied request."""
        self._log("search_start", iterations=self.iterations)
        rng = random.Random(self.seed)
        current = self._initial(matrix, start, end, start_minute, budget_s)
        self._move("initial_route", current)
        current = self._repair(
            matrix,
            current,
            start_minute,
            budget_s,
            rng,
        )
        best = current
        self._log_best("initial_solution", 0, best)
        destroy_weights = [1.0, 1.0]
        repair_weights = [1.0, 1.0]
        for iteration in range(self.iterations):
            self._iteration = iteration + 1
            destroy = rng.choices(range(2), weights=destroy_weights)[0]
            repair = rng.choices(range(2), weights=repair_weights)[0]
            count = rng.randint(1, max(1, min(4, len(current.route) - 2)))
            self._move("destroy", current,
                       destroy_operator="worst" if destroy else "random",
                       repair_operator="randomized" if repair else "greedy")
            partial = self._remove(
                matrix, current, start_minute, budget_s, rng, count, bool(destroy)
            )
            candidate = self._repair(
                matrix, partial, start_minute, budget_s, rng, bool(repair)
            )
            improved_best = self._quality(candidate) > self._quality(best)
            improved_current = self._quality(candidate) > self._quality(current)
            temperature = max(0.01, 1.0 - iteration / max(self.iterations, 1))
            delta = candidate.score - current.score
            accepted = improved_current or (
                delta < 0 and rng.random() < math.exp(delta / temperature)
            )
            if accepted:
                current = candidate
            if improved_best:
                best = candidate
            reward = 5.0 if improved_best else 2.0 if improved_current else 0.5
            destroy_weights[destroy] = 0.8 * destroy_weights[destroy] + 0.2 * reward
            repair_weights[repair] = 0.8 * repair_weights[repair] + 0.2 * reward
            self._log_best("iteration", iteration + 1, best, current=current.to_dict(),
                           candidate=candidate.to_dict(), accepted=accepted,
                           improved_best=improved_best, temperature=temperature)
        self._log_best("search_end", self.iterations, best)
        return best


class AntColony(_Search):
    """Search for a high-scoring route using ant colony optimisation.

    Each iteration builds a route from the same feasible starting route by
    randomly selecting feasible, improving POI insertions. Selection favours
    score gain relative to added travel time and edges with more pheromone.
    Pheromone evaporates after each iteration, then edges in the best route
    found receive reinforcement. Score is maximised first; travel time breaks
    score ties.

    Args:
        iterations: Number of routes to construct. Defaults to 100.
        seed: Seed for reproducible random choices. Defaults to 0.

    Raises:
        ValueError: If iterations is negative.
    """

    def __call__(
        self, matrix: ODMatrix, start: str, end: str, start_minute: int, budget_s: float
    ) -> RouteResult:
        self._log("search_start", iterations=self.iterations)
        rng = random.Random(self.seed)
        initial = self._initial(matrix, start, end, start_minute, budget_s)
        self._move("initial_route", initial)
        best = initial
        self._log_best("initial_solution", 0, best)
        pheromone: dict[tuple[str, str], float] = {}
        for iteration in range(self.iterations):
            self._iteration = iteration + 1
            current = initial
            self._move("ant_start", current)
            while True:
                options = []
                weights = []
                for poi in sorted(matrix.scores.keys() - set(current.route)):
                    for index in range(1, len(current.route)):
                        route = list(current.route)
                        route.insert(index, poi)
                        candidate = matrix.evaluate(route, start_minute, budget_s)
                        if candidate is None or self._quality(candidate) <= self._quality(current):
                            continue
                        gain = candidate.score - current.score
                        extra = max(candidate.duration_s - current.duration_s, 1.0)
                        trail = pheromone.get((route[index - 1], poi), 1.0)
                        trail *= pheromone.get((poi, route[index + 1]), 1.0)
                        options.append(candidate)
                        weights.append(trail * (1.0 + gain / extra))
                if not options:
                    break
                current = rng.choices(options, weights=weights, k=1)[0]
                self._move("insert", current)
            improved_best = self._quality(current) > self._quality(best)
            if improved_best:
                best = current
            for edge, trail in pheromone.items():
                pheromone[edge] = max(0.1, trail * 0.9)
            for edge in zip(best.route, best.route[1:]):
                pheromone[edge] = pheromone.get(edge, 1.0) + 1.0
            self._move("pheromone_update", best, best=best.to_dict(),
                       evaporation=0.9, pheromone_edges=len(pheromone))
            self._log_best("iteration", iteration + 1, best, current=current.to_dict(),
                           candidate=current.to_dict(), improved_best=improved_best)
        self._log_best("search_end", self.iterations, best)
        return best


class VNS(_Search):
    """Search for a high-scoring route with variable neighbourhood search.

    VNS starts with a feasible route and inserts promising POIs. Each
    iteration removes a number of stops, repairs the route, and tries moving
    selected stops or replacing them with unselected POIs. The number of
    stops removed cycles from one to four when no improvement is found and
    resets to one after an improvement. Score is maximised first; travel time
    breaks score ties.

    Args:
        iterations: Number of neighbourhood-search iterations. Defaults to 100.
        seed: Seed for reproducible random choices. Defaults to 0.

    Raises:
        ValueError: If iterations is negative.
    """

    def __call__(
        self, matrix: ODMatrix, start: str, end: str, start_minute: int, budget_s: float
    ) -> RouteResult:
        """Return the best feasible route found for the supplied request."""
        self._log("search_start", iterations=self.iterations)
        rng = random.Random(self.seed)
        best = self._initial(matrix, start, end, start_minute, budget_s)
        self._move("initial_route", best)
        best = self._repair(
            matrix,
            best,
            start_minute,
            budget_s,
            rng,
        )
        neighbourhood = 1
        self._log_best("initial_solution", 0, best)
        for iteration in range(self.iterations):
            self._iteration = iteration + 1
            self._move("shake", best, neighbourhood=neighbourhood)
            partial = self._remove(
                matrix, best, start_minute, budget_s, rng, neighbourhood, False
            )
            candidate = self._repair(matrix, partial, start_minute, budget_s, rng, True)
            candidate = self._local_search(matrix, candidate, start_minute, budget_s)
            accepted = self._quality(candidate) > self._quality(best)
            if accepted:
                best, neighbourhood = candidate, 1
            else:
                neighbourhood = neighbourhood % 4 + 1
            self._log_best("iteration", iteration + 1, best, current=best.to_dict(),
                           candidate=candidate.to_dict(), accepted=accepted,
                           improved_best=accepted, neighbourhood=neighbourhood)
        self._log_best("search_end", self.iterations, best)
        return best

    def _local_search(
        self, matrix: ODMatrix, current: RouteResult, minute: int, budget: float
    ) -> RouteResult:
        """Try every single-stop relocation and selected/unselected swap once."""
        best = current
        route = list(current.route)
        unused = sorted(matrix.scores.keys() - set(route))
        for index in range(1, len(route) - 1):
            for poi in unused:
                trial = route.copy()
                trial[index] = poi
                candidate = matrix.evaluate(trial, minute, budget)
                if candidate is not None and _Search._quality(
                    candidate
                ) > _Search._quality(best):
                    best = candidate
                    self._move("replace", best)
            shorter = route.copy()
            point = shorter.pop(index)
            for target in range(1, len(shorter)):
                trial = shorter.copy()
                trial.insert(target, point)
                candidate = matrix.evaluate(trial, minute, budget)
                if candidate is not None and _Search._quality(
                    candidate
                ) > _Search._quality(best):
                    best = candidate
                    self._move("relocate", best)
        return best
