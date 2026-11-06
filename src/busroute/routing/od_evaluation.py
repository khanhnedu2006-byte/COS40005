"""Shared OD baselines and evaluation for PPO experiments."""
from __future__ import annotations

import heapq
import time
from collections.abc import Callable

from busroute.routing.od_env import ODRouteEnv
from busroute.routing.orienteering import ODMatrix, RouteResult
from busroute.utils.timeslots import slot_of


def shortest_route(matrix: ODMatrix, start: str, end: str, minute: int, budget_s: float) -> RouteResult | None:
    """Find a fast route using time-dependent label search without waiting.

    Keep labels per point and arrival minute: earlier arrival does not always
    dominate later arrival when leg availability changes at a slot boundary.
    """
    queue = [(0.0, (start,))]
    best = {(start, minute): 0.0}
    while queue:
        elapsed, path = heapq.heappop(queue)
        current = path[-1]
        if current == end:
            return matrix.evaluate(path, minute, budget_s)
        slot = slot_of(int(minute + elapsed // 60) % 1440)
        for point in sorted(matrix.scores.keys() | {end}):
            if point in path:
                continue
            leg = matrix.legs.get((current, point, slot))
            if leg is None or elapsed + leg.duration_s > budget_s:
                continue
            new_elapsed = elapsed + leg.duration_s
            key = (point, int(minute + new_elapsed // 60))
            if new_elapsed < best.get(key, float('inf')):
                best[key] = new_elapsed
                heapq.heappush(queue, (new_elapsed, path + (point,)))
    return None


def greedy_route(matrix: ODMatrix, start: str, end: str, minute: int, budget_s: float) -> RouteResult | None:
    env = ODRouteEnv(matrix, start, end, minute, budget_s)
    while not env.done:
        mask = env.action_mask()
        options = [p for p, valid in mask.items() if valid and p != end]
        if options:
            def quality(p):
                leg = matrix.legs[env.current, p, env.slot()]
                new = (set(leg.passed) | {p}) - env.collected
                return sum(matrix.scores.get(q, 0) for q in new) / leg.duration_s
            action = max(options, key=quality)
        elif mask.get(end):
            action = end
        else:
            return None
        env.step(action)
    return matrix.evaluate(env.route, minute, budget_s) if env.current == end else None


def evaluate(matrix: ODMatrix, cases: list[tuple[str, str, int, float]],
             solvers: dict[str, Callable]) -> dict:
    """Record every case and aggregate success metrics, including failures."""
    report = {}
    for name, solver in solvers.items():
        rows = []
        for start, end, minute, budget in cases:
            began = time.perf_counter()
            error = None
            result = None
            try:
                proposed = solver(matrix, start, end, minute, budget)
                if proposed is not None and proposed.route[0] == start and proposed.route[-1] == end:
                    result = matrix.evaluate(proposed.route, minute, budget)
                if result is None:
                    error = "no valid route returned"
            except (ValueError, KeyError) as exc:
                error = str(exc)
            elapsed_ms = (time.perf_counter() - began) * 1000
            rows.append({"start": start, "end": end, "start_minute": minute, "budget_s": budget,
                         "completed": result is not None, "within_budget": result is not None,
                         "score": result.score if result else None,
                         "poi_count": len(result.collected) if result else None,
                         "travel_s": result.duration_s if result else None,
                         "runtime_ms": elapsed_ms, "route": list(result.route) if result else None,
                         "error": error})
        successful = [r for r in rows if r["completed"]]
        def mean(key, items):
            return sum(r[key] for r in items) / len(items) if items else None
        report[name] = {"cases": len(rows), "completed": len(successful),
                        "completion_rate": len(successful) / len(rows) if rows else 0,
                        "budget_compliance_rate": len(successful) / len(rows) if rows else 0,
                        "mean_score_successes": mean("score", successful),
                        "mean_poi_count_successes": mean("poi_count", successful),
                        "mean_travel_s_successes": mean("travel_s", successful),
                        "mean_runtime_ms": mean("runtime_ms", rows), "results": rows}
    return report
