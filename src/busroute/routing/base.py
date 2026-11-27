"""Shared solver base classes and standardized input/output contracts for routing."""

from __future__ import annotations

import math
import multiprocessing as mp
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

from busroute.utils.timeslots import to_minutes

TContext = TypeVar("TContext")
TRawResult = TypeVar("TRawResult")

HHMM_RE = re.compile(r"[0-2][0-9]:[0-5][0-9]")


def _never_cancelled() -> bool:
    return False


def parse_departure_minute(departure: str) -> int:
    """Validate an ``HH:MM`` local departure string and return minutes after midnight."""
    if (
        not isinstance(departure, str)
        or not HHMM_RE.fullmatch(departure)
        or int(departure[:2]) >= 24
    ):
        raise ValueError("Departure must be a valid HH:MM time")
    return to_minutes(departure)


def format_minute_hhmm(start_minute: int) -> str:
    """Format a validated minute-of-day ``[0, 1440)`` as ``HH:MM``."""
    if (
        isinstance(start_minute, bool)
        or not isinstance(start_minute, int)
        or not 0 <= start_minute < 1440
    ):
        raise ValueError("Departure must be a minute within the day")
    return f"{start_minute // 60:02d}:{start_minute % 60:02d}"


@dataclass(frozen=True)
class RoadDatasetSpec:
    """Standardized input dataset descriptor for graph-based route solvers."""

    nodes_path: Path
    edges_path: Path
    widths_path: Path
    points: tuple[dict[str, Any], ...]
    reduced_dir: Path
    use_reduced: bool = True

    @classmethod
    def from_raw(
        cls,
        nodes_path: str | Path,
        edges_path: str | Path,
        widths_path: str | Path,
        points: Sequence[dict[str, Any]],
        *,
        reduced_dir: str | Path,
        use_reduced: bool = True,
    ) -> RoadDatasetSpec:
        """Normalize file paths and point metadata into a frozen ``RoadDatasetSpec``."""
        return cls(
            nodes_path=Path(nodes_path),
            edges_path=Path(edges_path),
            widths_path=Path(widths_path),
            points=tuple(points),
            reduced_dir=Path(reduced_dir),
            use_reduced=use_reduced,
        )


@dataclass(frozen=True)
class RouteRequest:
    """Standardized routing request shared across all POI and street-level solvers."""

    start: str
    end: str
    budget_min: float
    start_minute: int = 480
    iterations: int = 100
    seed: int = 0
    workers: int = 1
    safety_margin: float = 1.05
    cancelled: Callable[[], bool] = _never_cancelled

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError("Start and end must be distinct")
        if not math.isfinite(self.budget_min) or self.budget_min <= 0:
            raise ValueError("Time budget must be positive and finite")
        format_minute_hhmm(self.start_minute)
        if isinstance(self.iterations, bool) or self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        if (
            isinstance(self.workers, bool)
            or not isinstance(self.workers, int)
            or self.workers < 1
        ):
            raise ValueError("workers must be a positive integer")
        if not math.isfinite(self.safety_margin) or self.safety_margin < 1.0:
            raise ValueError("safety_margin must be finite and >= 1.0")

    @property
    def budget_s(self) -> float:
        """Total travel-time budget in seconds."""
        return float(self.budget_min) * 60.0

    @property
    def departure(self) -> str:
        """Local departure time formatted as ``HH:MM``."""
        return format_minute_hhmm(self.start_minute)

    def check_cancelled(self) -> None:
        """Raise ``InterruptedError`` when cooperative cancellation is requested."""
        if self.cancelled():
            raise InterruptedError("Search cancelled")

    @classmethod
    def from_departure(
        cls,
        start: str,
        end: str,
        budget_min: float,
        *,
        departure: str = "08:00",
        iterations: int = 100,
        seed: int = 0,
        workers: int = 1,
        safety_margin: float = 1.05,
        cancelled: Callable[[], bool] = _never_cancelled,
    ) -> RouteRequest:
        """Build a ``RouteRequest`` from an ``HH:MM`` departure string."""
        return cls(
            start=start,
            end=end,
            budget_min=budget_min,
            start_minute=parse_departure_minute(departure),
            iterations=iterations,
            seed=seed,
            workers=workers,
            safety_margin=safety_margin,
            cancelled=cancelled,
        )


@dataclass(frozen=True)
class RouteOutput:
    """Standardized solver output convertible to the shared ``RouteSnapshot`` JSON contract."""

    route: tuple[str, ...]
    score: float
    distance_km: float
    duration_min: float
    collected: tuple[str, ...]
    geometry: dict[str, Any]
    narrow_m: float
    unknown_width_m: float
    instructions: tuple[dict[str, Any], ...]

    def to_snapshot(self) -> dict[str, Any]:
        """Serialize to the common ``RouteSnapshot`` dictionary format."""
        return {
            "route": list(self.route),
            "score": float(self.score),
            "distance_km": float(self.distance_km),
            "duration_min": float(self.duration_min),
            "collected": list(self.collected),
            "geometry": self.geometry,
            "narrow_m": float(self.narrow_m),
            "unknown_width_m": float(self.unknown_width_m),
            "instructions": list(self.instructions),
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> RouteOutput:
        """Construct a ``RouteOutput`` from an existing ``RouteSnapshot`` dictionary."""
        return cls(
            route=tuple(snapshot["route"]),
            score=float(snapshot["score"]),
            distance_km=float(snapshot["distance_km"]),
            duration_min=float(snapshot["duration_min"]),
            collected=tuple(snapshot.get("collected", ())),
            geometry=dict(snapshot["geometry"]),
            narrow_m=float(snapshot.get("narrow_m", 0.0)),
            unknown_width_m=float(snapshot.get("unknown_width_m", 0.0)),
            instructions=tuple(snapshot.get("instructions", ())),
        )


def _fork_worker_dispatch(payload: tuple[type[Any], Any]) -> Any:
    """Pickle-safe top-level dispatcher delegating ``mp.Pool`` tasks to solver classes."""
    solver_cls, task = payload
    return solver_cls._fork_worker(task)


class BaseRouteSolver(ABC, Generic[TContext, TRawResult]):
    """Abstract base class defining ``parse_data -> solve -> format_output`` for solvers."""

    name: ClassVar[str] = "base"
    mode: ClassVar[str] = "base"

    def run(
        self,
        request: RouteRequest,
        *,
        data: RoadDatasetSpec | None = None,
        context: TContext | None = None,
    ) -> RouteOutput:
        """Parse inputs if needed, run the algorithm, and format the standardized output."""
        request.check_cancelled()
        if context is None:
            if data is None:
                raise ValueError("Either data or context must be provided")
            context = self.parse_data(data, request)
        raw = self.solve(context, request)
        return self.format_output(context, raw, request)

    def solve_snapshot(
        self,
        request: RouteRequest,
        *,
        data: RoadDatasetSpec | None = None,
        context: TContext | None = None,
    ) -> dict[str, Any]:
        """Run the solver pipeline and return a JSON-compatible ``RouteSnapshot`` dict."""
        return self.run(request, data=data, context=context).to_snapshot()

    @abstractmethod
    def parse_data(self, data: RoadDatasetSpec, request: RouteRequest) -> TContext:
        """Load and parse raw road/POI files into the solver's execution context."""

    @abstractmethod
    def solve(self, context: TContext, request: RouteRequest) -> TRawResult:
        """Execute the routing algorithm on the prepared context."""

    @abstractmethod
    def format_output(
        self, context: TContext, result: TRawResult, request: RouteRequest
    ) -> RouteOutput:
        """Convert the algorithm's native result into a standardized ``RouteOutput``."""

    @staticmethod
    def _can_fork(workers: int, iterations: int) -> bool:
        return workers > 1 and iterations > 1 and "fork" in mp.get_all_start_methods()
