"""Time-dependent orienteering over the project's point-to-point OD matrix.

The matrix describes a single route per pair and departure slot. It does not
resolve restrictions encountered partway through a leg or include visit times.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from busroute.utils.timeslots import slot_of


@dataclass(frozen=True)
class Leg:
    """Travel time in seconds and POIs passed by one feasible OD leg."""

    duration_s: float
    passed: frozenset[str]


@dataclass(frozen=True)
class RouteResult:
    """A feasible route, its total travel time, and its unique collected POIs."""

    route: tuple[str, ...]
    score: float
    duration_s: float
    collected: frozenset[str]

    def to_dict(self) -> dict:
        """Return the common JSON-compatible representation of a route."""
        return {"route": list(self.route), "score": self.score,
                "duration_min": self.duration_s / 60, "collected": sorted(self.collected)}


class ODMatrix:
    """Directed, slot-specific OD legs and the scores of candidate POIs."""

    def __init__(self, legs: dict[tuple[str, str, str], Leg], scores: dict[str, float],
                 points: set[str]) -> None:
        self.legs = legs
        self.scores = scores
        self.points = points

    @classmethod
    def from_csv(cls, path: str | Path) -> ODMatrix:
        """Load a UTF-8 OD matrix, retaining only rows marked feasible.

        A missing or infeasible leg is not traversable.

        Args:
            path: Path to the CSV matrix, with or without a UTF-8 BOM.

        Returns:
            A matrix containing feasible legs, POI scores, and known points.

        Raises:
            ValueError: If rows are duplicated, values are invalid or
                inconsistent, passed POIs are unknown, or the matrix is empty.
        """
        legs: dict[tuple[str, str, str], Leg] = {}
        scores: dict[str, float] = {}
        points: set[str] = set()
        seen: set[tuple[str, str, str]] = set()
        with open(path, encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                origin, dest, slot = row["from_id"], row["to_id"], row["slot"]
                key = (origin, dest, slot)
                if key in seen:
                    raise ValueError(f"Duplicate OD row: {key}")
                seen.add(key)
                points.update((origin, dest))
                score = float(row["to_poi_score"])
                if not math.isfinite(score) or score < 0:
                    raise ValueError(f"Invalid POI score for {dest}: {score}")
                if row["to_type"] == "poi":
                    if dest in scores and scores[dest] != score:
                        raise ValueError(f"Inconsistent POI score for {dest}")
                    scores[dest] = score
                feasible = row["feasible"]
                if feasible not in ("0", "1"):
                    raise ValueError(f"Invalid feasibility for {key}: {feasible}")
                if feasible == "1":
                    duration = float(row["duration_s"])
                    if not math.isfinite(duration) or duration <= 0:
                        raise ValueError(f"Invalid duration for {key}: {duration}")
                    passed = frozenset(p.strip() for p in row["pois_passed"].split(";") if p.strip())
                    legs[key] = Leg(duration, passed)
        if not points:
            raise ValueError("OD matrix is empty")
        unknown = set().union(*(leg.passed for leg in legs.values())) - scores.keys()
        if unknown:
            raise ValueError(f"Unknown passed POIs: {sorted(unknown)}")
        return cls(legs, scores, points)

    def evaluate(self, route: Sequence[str], start_minute: int, budget_s: float) -> RouteResult | None:
        """Evaluate a route; return None if a leg or the time budget is infeasible.

        A start POI counts as collected. Passing a POI and stopping at it earn
        the same score once. A leg uses its departure slot, even across a slot
        boundary. No waiting or dwell time is modelled.

        Args:
            route: Ordered IDs of at least two known points.
            start_minute: Departure minute after midnight, in [0, 1440).
            budget_s: Positive finite travel-time budget in seconds.

        Returns:
            The route result, or None if a leg is unavailable or exceeds the
            budget.

        Raises:
            ValueError: If the route, departure minute, or budget is invalid.
        """
        if len(route) < 2 or any(point not in self.points for point in route):
            raise ValueError("Route must contain at least two known points")
        if not 0 <= start_minute < 1440 or not math.isfinite(budget_s) or budget_s <= 0:
            raise ValueError("start_minute must be in [0, 1440), and budget_s must be positive")
        elapsed = 0.0
        collected = {route[0]} & self.scores.keys()
        for origin, dest in zip(route, route[1:]):
            slot = slot_of(int(start_minute + elapsed // 60) % 1440)
            leg = self.legs.get((origin, dest, slot))
            if leg is None:
                return None
            elapsed += leg.duration_s
            if elapsed > budget_s:
                return None
            collected.update(leg.passed)
            if dest in self.scores:
                collected.add(dest)
        return RouteResult(tuple(route), sum(self.scores[p] for p in sorted(collected)), elapsed, frozenset(collected))
