"""ALNS, VNS and ACO on time-budgeted, graph-derived POI connections."""

from __future__ import annotations

import math
import random
from itertools import pairwise


def _insertions(problem, current):
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


def _repair(problem, initial, rng, randomized=False):
    current = initial
    while True:
        options = list(_insertions(problem, current))
        if not options:
            return current
        options.sort(key=lambda item: (item[0], item[1].quality), reverse=True)
        current = rng.choice(options[:3])[1] if randomized else options[0][1]


def _remove(problem, current, rng, count, worst=False):
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


def _local_search(problem, current):
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


def _ant_colony(problem, best, rng, iterations):
    pheromone = {}
    for _ in range(iterations):
        problem.graph.check_cancelled()
        current = problem.direct
        while options := list(_insertions(problem, current)):
            weights = []
            for gain_per_second, candidate, index in options:
                before, poi, after = candidate.route[index - 1 : index + 2]
                trail_in = pheromone.get((before, poi), 1.0)
                trail_out = pheromone.get((poi, after), 1.0)
                weights.append(trail_in * trail_out * (1.0 + gain_per_second))
            _, current, _ = rng.choices(options, weights=weights, k=1)[0]
        if current.quality > best.quality:
            best = current
        for connection, trail in pheromone.items():
            pheromone[connection] = max(0.1, trail * 0.9)
        for connection in pairwise(best.route):
            pheromone[connection] = pheromone.get(connection, 1.0) + 1.0
    return best


def search(problem, algorithm, *, iterations=100, seed=0):
    """Return the best feasible route. Connections reuse cached trees for each departure slot."""
    if algorithm not in ("alns", "vns", "aco"):
        raise ValueError("Choose alns, vns or aco")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    rng = random.Random(seed)
    current = best = _repair(problem, problem.direct, rng)
    if algorithm == "aco":
        return _ant_colony(problem, best, rng, iterations)
    destroy_weights, repair_weights = [1.0, 1.0], [1.0, 1.0]
    neighbourhood = 1
    for iteration in range(iterations):
        problem.graph.check_cancelled()
        if algorithm == "alns":
            destroy = rng.choices(range(2), weights=destroy_weights)[0]
            repair = rng.choices(range(2), weights=repair_weights)[0]
            count = rng.randint(1, max(1, min(4, len(current.route) - 2)))
            partial = _remove(problem, current, rng, count, bool(destroy))
            candidate = _repair(problem, partial, rng, bool(repair))
            improved_best = candidate.quality > best.quality
            improved_current = candidate.quality > current.quality
            temperature = max(0.01, 1.0 - iteration / max(iterations, 1))
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
        else:
            partial = _remove(problem, best, rng, neighbourhood)
            candidate = _local_search(problem, _repair(problem, partial, rng, True))
            if candidate.quality > best.quality:
                best, neighbourhood = candidate, 1
            else:
                neighbourhood = neighbourhood % 4 + 1
    return best
