"""Basic ALNS and VNS search for the time-dependent orienteering problem."""
from __future__ import annotations

import math
import random

from busroute.routing.orienteering import ODMatrix, RouteResult


def _quality(result: RouteResult) -> tuple[float, float]:
    """Maximise score first; prefer a shorter route only when scores tie."""
    return result.score, -result.duration_s


def _initial(matrix: ODMatrix, start: str, end: str, minute: int, budget: float) -> RouteResult:
    if start == end or start not in matrix.points or end not in matrix.points:
        raise ValueError("Start and end must be distinct known points")
    direct = matrix.evaluate([start, end], minute, budget)
    if direct is not None:
        return direct
    # A direct leg can be unavailable even when a one-stop route is possible.
    candidates = (matrix.evaluate([start, p, end], minute, budget)
                  for p in sorted(matrix.scores) if p not in (start, end))
    feasible = [result for result in candidates if result is not None]
    if not feasible:
        raise ValueError("No feasible direct or one-stop initial route found; a longer feasible route may exist")
    return max(feasible, key=_quality)


def _repair(matrix: ODMatrix, initial: RouteResult, minute: int, budget: float,
            rng: random.Random, randomized: bool = False) -> RouteResult:
    """Repeatedly insert an unselected stop when it improves route quality."""
    current = initial
    while True:
        options: list[tuple[float, RouteResult]] = []
        for poi in sorted(matrix.scores.keys() - set(current.route)):
            for index in range(1, len(current.route)):
                route = list(current.route)
                route.insert(index, poi)
                candidate = matrix.evaluate(route, minute, budget)
                if candidate is not None and _quality(candidate) > _quality(current):
                    gain = candidate.score - current.score
                    extra = max(candidate.duration_s - current.duration_s, 1.0)
                    options.append((gain / extra, candidate))
        if not options:
            return current
        options.sort(key=lambda item: (item[0], *_quality(item[1])), reverse=True)
        current = rng.choice(options[:min(3, len(options))])[1] if randomized else options[0][1]


def _remove(matrix: ODMatrix, current: RouteResult, minute: int, budget: float,
            rng: random.Random, count: int, worst: bool) -> RouteResult:
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
        current = max(options, key=_quality) if worst else rng.choice(options)
    return current


def _check(iterations: int) -> None:
    if iterations < 0:
        raise ValueError("iterations must be non-negative")


def alns(matrix: ODMatrix, start: str, end: str, start_minute: int, budget_s: float,
         *, iterations: int = 100, seed: int = 0) -> RouteResult:
    """Run adaptive destroy/repair search and return the best feasible route.

    Random and low-contribution removal, and greedy and randomized insertion,
    receive adaptive weights. Simulated annealing permits score decreases in
    the current solution while the best solution is retained separately.

    Args:
        matrix: Directed, slot-specific OD legs and POI scores.
        start: Starting point ID.
        end: Destination point ID, distinct from start.
        start_minute: Departure minute after midnight.
        budget_s: Travel-time budget in seconds.
        iterations: Number of search iterations; must be non-negative.
        seed: Seed for the local random number generator.

    Returns:
        The best feasible route found, prioritizing score then shorter duration.

    Raises:
        ValueError: If inputs are invalid or no direct or one-stop initial
            route is feasible. A longer feasible route may still exist.
    """
    _check(iterations)
    rng = random.Random(seed)
    current = _repair(matrix, _initial(matrix, start, end, start_minute, budget_s), start_minute, budget_s, rng)
    best = current
    destroy_weights = [1.0, 1.0]
    repair_weights = [1.0, 1.0]
    for iteration in range(iterations):
        destroy = rng.choices(range(2), weights=destroy_weights)[0]
        repair = rng.choices(range(2), weights=repair_weights)[0]
        count = rng.randint(1, max(1, min(4, len(current.route) - 2)))
        partial = _remove(matrix, current, start_minute, budget_s, rng, count, bool(destroy))
        candidate = _repair(matrix, partial, start_minute, budget_s, rng, bool(repair))
        improved_best = _quality(candidate) > _quality(best)
        improved_current = _quality(candidate) > _quality(current)
        temperature = max(0.01, 1.0 - iteration / max(iterations, 1))
        delta = candidate.score - current.score
        if improved_current or (delta < 0 and rng.random() < math.exp(delta / temperature)):
            current = candidate
        if improved_best:
            best = candidate
        reward = 5.0 if improved_best else 2.0 if improved_current else 0.5
        destroy_weights[destroy] = 0.8 * destroy_weights[destroy] + 0.2 * reward
        repair_weights[repair] = 0.8 * repair_weights[repair] + 0.2 * reward
    return best


def vns(matrix: ODMatrix, start: str, end: str, start_minute: int, budget_s: float,
        *, iterations: int = 100, seed: int = 0) -> RouteResult:
    """Run variable-neighbourhood search using removal, swap and relocation.

    Increasing removal counts shake the incumbent; insertion and local moves
    rebuild and improve it. Only feasible solutions are returned.

    Args:
        matrix: Directed, slot-specific OD legs and POI scores.
        start: Starting point ID.
        end: Destination point ID, distinct from start.
        start_minute: Departure minute after midnight.
        budget_s: Travel-time budget in seconds.
        iterations: Number of search iterations; must be non-negative.
        seed: Seed for the local random number generator.

    Returns:
        The best feasible route found, prioritizing score then shorter duration.

    Raises:
        ValueError: If inputs are invalid or no direct or one-stop initial
            route is feasible. A longer feasible route may still exist.
    """
    _check(iterations)
    rng = random.Random(seed)
    best = _repair(matrix, _initial(matrix, start, end, start_minute, budget_s), start_minute, budget_s, rng)
    neighbourhood = 1
    for _ in range(iterations):
        partial = _remove(matrix, best, start_minute, budget_s, rng, neighbourhood, False)
        candidate = _repair(matrix, partial, start_minute, budget_s, rng, True)
        candidate = _local_search(matrix, candidate, start_minute, budget_s)
        if _quality(candidate) > _quality(best):
            best, neighbourhood = candidate, 1
        else:
            neighbourhood = neighbourhood % 4 + 1
    return best


def _local_search(matrix: ODMatrix, current: RouteResult, minute: int, budget: float) -> RouteResult:
    """Try every single-stop relocation and selected/unselected swap once."""
    best = current
    route = list(current.route)
    unused = sorted(matrix.scores.keys() - set(route))
    for index in range(1, len(route) - 1):
        for poi in unused:
            trial = route.copy()
            trial[index] = poi
            candidate = matrix.evaluate(trial, minute, budget)
            if candidate is not None and _quality(candidate) > _quality(best):
                best = candidate
        shorter = route.copy()
        point = shorter.pop(index)
        for target in range(1, len(shorter)):
            trial = shorter.copy()
            trial.insert(target, point)
            candidate = matrix.evaluate(trial, minute, budget)
            if candidate is not None and _quality(candidate) > _quality(best):
                best = candidate
    return best
