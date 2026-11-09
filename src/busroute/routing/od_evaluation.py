"""Shared OD baselines and evaluation for PPO experiments."""
from __future__ import annotations

import heapq
import time
from collections.abc import Callable

from busroute.routing.od_env import ODRouteEnv
from busroute.routing.orienteering import ODMatrix, RouteResult
from busroute.utils.timeslots import slot_of


def shortest_route(matrix: ODMatrix, start: str, end: str, minute: int, budget_s: float,
                   *, progress: Callable | None = None, trace: bool = False) -> RouteResult | None:
    """Find a fast route with a Dijkstra-style time-dependent label search.

    Expand paths in order of elapsed travel time, using the departure slot of
    each leg and discarding paths that exceed the budget. Labels distinguish
    arrival minutes because availability can change between slots. This
    approximation does not model waiting and does not guarantee an exact
    shortest route for time-dependent travel times.

    Args:
        matrix: Directed, time-dependent OD matrix.
        start: Starting point ID.
        end: Destination point ID.
        minute: Departure time in minutes after midnight.
        budget_s: Maximum travel time in seconds.
        progress: Optional callback for search lifecycle and progress records.
        trace: Also emit each expanded partial path for map playback. Defaults
            to False to preserve the lightweight command-line progress format.

    Returns:
        The first feasible destination route found, or None if the search
        exhausts its candidates without reaching the destination.
    """
    queue = [(0.0, (start,))]
    best = {(start, minute): 0.0}
    expanded = 0
    if progress is not None:
        progress({"event": "search_start"})
    while queue:
        elapsed, path = heapq.heappop(queue)
        expanded += 1
        current = path[-1]
        if trace and progress is not None:
            partial = (matrix.evaluate(path, minute, budget_s) if len(path) > 1 else
                       RouteResult(path, matrix.scores.get(start, 0), 0,
                                   frozenset({start} & matrix.scores.keys())))
            if partial is not None:
                progress({"event": "operator", "action": "expand", "expanded": expanded,
                          "queued": len(queue), "candidate": partial.to_dict()})
        if current == end:
            result = matrix.evaluate(path, minute, budget_s)
            if progress is not None:
                completion = {"event": "search_end", "expanded": expanded, "found": result is not None,
                              "duration_min": result.duration_s / 60 if result is not None else None}
                if trace and result is not None:
                    completion.update(best=result.to_dict(), current=result.to_dict())
                progress(completion)
            return result
        if progress is not None and expanded % 100 == 0:
            progress({"event": "search_progress", "expanded": expanded, "queued": len(queue)})
        slot = slot_of(int(minute + elapsed // 60) % 1440)
        for point in sorted(matrix.points):
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
    if progress is not None:
        progress({"event": "search_end", "expanded": expanded, "found": False, "duration_min": None})
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
