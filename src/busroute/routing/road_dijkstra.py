"""Shortest road route with a conservative, name-based narrow-road penalty."""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from busroute.utils.geo import normalize_road_name

if TYPE_CHECKING:
    from Web.types import AlgorithmName, PointMetadata, RouteSnapshot


def road_widths(path: str | Path) -> dict[str, float]:
    """Load unambiguous numeric road widths from the cleaned survey CSV.

    Args:
        path: UTF-8 CSV with the Vietnamese road-name and carriageway-width
            columns. Decimal commas are accepted for a single positive width.

    Returns:
        Normalized road names mapped to widths in metres. Conflicting,
        missing or nonnumeric widths are excluded instead of guessed.

    Raises:
        OSError: The CSV cannot be opened.
        KeyError: A required CSV column is missing.
    """
    values: defaultdict[str, set[float | None]] = defaultdict(set)
    with open(path, encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            name = normalize_road_name(row["Tên đường"])
            raw = row["Chiều rộng mặt (m)"].strip().replace(",", ".")
            if not name:
                continue
            if not re.fullmatch(r"\d+(?:\.\d+)?", raw) or float(raw) <= 0:
                values[name].add(None)
            else:
                values[name].add(float(raw))
    return {
        name: cast(float, next(iter(widths)))
        for name, widths in values.items()
        if len(widths) == 1 and None not in widths
    }


def road_instructions(
    coords: Sequence[Sequence[float]],
    details: Iterable[tuple[float, float | None, str]],
    scale: NDArray[np.float64],
) -> list[dict[str, Any]]:
    """Group road edges and estimate turn directions from node chords.

    Args:
        coords: Longitude/latitude points, one more than the edge-detail count.
        details: Per-edge tuples of length in metres, optional width in metres,
            and display road name. Consecutive equal names form one instruction.
        scale: Two-axis longitude/latitude-to-metre scale, shape (2,).

    Returns:
        JSON instruction dictionaries with action, road, distance_m, turn
        coordinate and the road group's coordinates. Directions are estimates
        because base-table edges contain node chords rather than road curves.
    """
    steps: list[dict[str, Any]] = []
    for index, (length, _width, name) in enumerate(details):
        if steps and steps[-1]["road"] == name:
            steps[-1]["distance_m"] += length
            steps[-1]["coordinates"].append(coords[index + 1])
            continue
        action = "Start on" if index == 0 else "Continue onto"
        if index:
            incoming = (np.asarray(coords[index]) - coords[index - 1]) * scale
            outgoing = (np.asarray(coords[index + 1]) - coords[index]) * scale
            if np.linalg.norm(incoming) > 1 and np.linalg.norm(outgoing) > 1:
                cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
                angle = math.degrees(
                    math.atan2(cross, float(np.dot(incoming, outgoing)))
                )
                if abs(angle) >= 150:
                    action = "Make a U-turn onto"
                elif angle >= 30:
                    action = "Turn left onto"
                elif angle <= -30:
                    action = "Turn right onto"
        steps.append(
            {
                "action": action,
                "road": name,
                "distance_m": length,
                "coordinate": coords[index],
                "coordinates": [coords[index], coords[index + 1]],
            }
        )
    return steps


def find_route(
    nodes_path: str | Path,
    edges_path: str | Path,
    widths_path: str | Path,
    points: Sequence[PointMetadata],
    start: str,
    end: str,
    cancelled: Callable[[], bool] = lambda: False,
    *,
    algorithm: AlgorithmName = "dijkstra",
    budget_min: float = 240.0,
    departure: str = "08:00",
    iterations: int = 100,
    seed: int = 0,
    checkpoint: str | Path | None = None,
) -> RouteSnapshot:
    """Solve a time-budgeted route on the complete directed road graph.

    Args:
        nodes_path: Base-node Parquet file.
        edges_path: Directed-edge Parquet file.
        widths_path: Cleaned road-width survey CSV.
        points: Configured hub/POI metadata, including coordinates and scores.
        start: Configured starting point ID.
        end: Distinct configured destination point ID.
        cancelled: Cooperative cancellation predicate; True requests a stop.
        algorithm: One of dijkstra, alns, vns, aco or ppo.
        budget_min: Positive finite travel-time budget in minutes.
        departure: Local departure time in HH:MM format.
        iterations: ALNS, VNS and ACO iteration budget.
        seed: ALNS, VNS and ACO sampling seed.
        checkpoint: Required SB3 ZIP path for the selected PPO solver.

    Returns:
        Shared route snapshot with actual edge geometry, instructions, unique
        POI score, passed POIs, minutes, kilometres and width-coverage lengths.

    Raises:
        OSError: A graph, survey or checkpoint file cannot be read.
        ValueError: Inputs/checkpoint are invalid or no feasible route is found.
        InterruptedError: The cancellation predicate requests a stop.
    """
    from busroute.routing.road_graph import RoadGraph, RoadProblem
    from busroute.routing.road_metaheuristics import search
    from busroute.utils.timeslots import to_minutes

    if (
        not re.fullmatch(r"[0-2][0-9]:[0-5][0-9]", departure)
        or int(departure[:2]) >= 24
    ):
        raise ValueError("Departure must be a valid HH:MM time")
    if algorithm == "ppo":
        if not checkpoint:
            raise ValueError("Choose a trained checkpoint for the selected PPO solver")
        from busroute.routing.ppo_train import infer_route

        return cast(
            "RouteSnapshot",
            infer_route(
                nodes_path,
                edges_path,
                widths_path,
                points,
                start,
                end,
                checkpoint,
                budget_min=budget_min,
                start_minute=to_minutes(departure),
                cancelled=cancelled,
            ),
        )
    graph = RoadGraph.from_files(nodes_path, edges_path, widths_path, points, cancelled)
    problem = RoadProblem(
        graph,
        start,
        end,
        budget_min,
        start_minute=to_minutes(departure),
        all_pois=algorithm != "dijkstra",
    )
    result = (
        problem.direct
        if algorithm == "dijkstra"
        else search(problem, algorithm, iterations=iterations, seed=seed)
    )
    return cast("RouteSnapshot", graph.snapshot(problem, result))
