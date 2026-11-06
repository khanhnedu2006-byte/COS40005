"""Basic ALNS and VNS search for the time-dependent orienteering problem."""

from __future__ import annotations

import math
import random
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
    """

    def __init__(self, *, iterations: int = 100, seed: int = 0) -> None:
        self._check(iterations)
        self.iterations = iterations
        self.seed = seed

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

    @staticmethod
    def _repair(
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

    @staticmethod
    def _remove(
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
        return current

    @staticmethod
    def _check(iterations: int) -> None:
        """Reject a negative iteration count."""
        if iterations < 0:
            raise ValueError("iterations must be non-negative")


class ALNS(_Search):
    """Adaptive destroy/repair search with simulated annealing."""

    def __call__(
        self, matrix: ODMatrix, start: str, end: str, start_minute: int, budget_s: float
    ) -> RouteResult:
        """Return the best feasible route found for the supplied request."""
        rng = random.Random(self.seed)
        current = self._repair(
            matrix,
            self._initial(matrix, start, end, start_minute, budget_s),
            start_minute,
            budget_s,
            rng,
        )
        best = current
        destroy_weights = [1.0, 1.0]
        repair_weights = [1.0, 1.0]
        for iteration in range(self.iterations):
            destroy = rng.choices(range(2), weights=destroy_weights)[0]
            repair = rng.choices(range(2), weights=repair_weights)[0]
            count = rng.randint(1, max(1, min(4, len(current.route) - 2)))
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


class VNS(_Search):
    """Variable-neighbourhood search with removal, swaps and relocation."""

    def __call__(
        self, matrix: ODMatrix, start: str, end: str, start_minute: int, budget_s: float
    ) -> RouteResult:
        """Return the best feasible route found for the supplied request."""
        rng = random.Random(self.seed)
        best = self._repair(
            matrix,
            self._initial(matrix, start, end, start_minute, budget_s),
            start_minute,
            budget_s,
            rng,
        )
        neighbourhood = 1
        for _ in range(self.iterations):
            partial = self._remove(
                matrix, best, start_minute, budget_s, rng, neighbourhood, False
            )
            candidate = self._repair(matrix, partial, start_minute, budget_s, rng, True)
            candidate = self._local_search(matrix, candidate, start_minute, budget_s)
            if self._quality(candidate) > self._quality(best):
                best, neighbourhood = candidate, 1
            else:
                neighbourhood = neighbourhood % 4 + 1
        return best

    @staticmethod
    def _local_search(
        matrix: ODMatrix, current: RouteResult, minute: int, budget: float
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
        return best
