"""Local web API, isolated solver processes, and reusable OSM geometry."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import multiprocessing as mp
import queue
import runpy
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue as ProcessQueue
from multiprocessing.synchronize import Event as ProcessEvent
from pathlib import Path
from typing import TypeAlias, cast

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from busroute.routing.metaheuristics import ALNS, VNS, AntColony
from busroute.routing.od_evaluation import shortest_route
from busroute.routing.orienteering import ODMatrix, RouteResult
from busroute.utils.timeslots import to_minutes
from Web.road_paths import RoadPaths, load_points
from Web.types import (
    AlgorithmName,
    BootstrapPayload,
    GeometryLeg,
    GeometryPayload,
    GraphStatus,
    GraphStatusPayload,
    JSONObject,
    PointMetadata,
    RouteSnapshot,
    RunCreatedPayload,
    RunStatus,
    RunStatusPayload,
    RunStoppedPayload,
    SearchParameters,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name("assets")
TERMINAL = {"result", "error", "stopped"}
logger = logging.getLogger(__name__)
GraphLoader: TypeAlias = Callable[[list[PointMetadata]], RoadPaths]


class RunRequest(BaseModel):
    """Validate route inputs before spawning a search process.

    Attributes:
        algorithm: ALNS, VNS, ACO, or the Dijkstra-style travel-time baseline.
        start: Existing starting point ID.
        end: Existing destination ID, distinct from start.
        departure: Local HH:MM departure time.
        budget_min: Positive finite travel-time budget in minutes.
        iterations: Non-negative iteration count; unused by Dijkstra.
        seed: Integer random seed; unused by Dijkstra.
    """

    model_config = ConfigDict(extra="forbid")
    algorithm: AlgorithmName = "alns"
    start: str
    end: str
    departure: str = Field(default="08:00", pattern=r"^\d{2}:\d{2}$")
    budget_min: float = Field(default=240, gt=0, allow_inf_nan=False)
    iterations: int = Field(default=100, ge=0, strict=True)
    seed: int = Field(default=0, strict=True)

    @model_validator(mode="after")
    def validate_route(self) -> RunRequest:
        """Check endpoint distinction and the departure clock range.

        Returns:
            This validated request.

        Raises:
            ValueError: Endpoints match or the departure time is invalid.
        """
        if self.start == self.end:
            raise ValueError("Choose different start and destination points")
        hour, minute = map(int, self.departure.split(":"))
        if hour >= 24 or minute >= 60:
            raise ValueError("Departure must be a valid time in HH:MM")
        return self


class GeometryRequest(BaseModel):
    """Request at most 200 directed point-pair geometries.

    Attributes:
        legs: Origin/destination ID pairs, deduplicated by the endpoint.
    """

    legs: list[tuple[str, str]] = Field(max_length=200)


class SearchCancelled(Exception):
    """Stop a worker cooperatively after its cancellation flag is set."""


def route_snapshot(result: RouteResult) -> RouteSnapshot:
    """Serialize a domain result into the typed browser snapshot contract.

    Args:
        result: Completed feasible route evaluated on the OD matrix.

    Returns:
        A JSON-compatible route, score, duration, and collected-POI record.
    """
    return {
        "route": list(result.route),
        "score": result.score,
        "duration_min": result.duration_s / 60,
        "collected": sorted(result.collected),
    }


def search_worker(
    matrix_path: str,
    params: SearchParameters,
    output: ProcessQueue[RunStatusPayload],
    cancelled: ProcessEvent,
) -> None:
    """Compute a final route without recording intermediate visualization data.

    Args:
        matrix_path: Absolute path to the selected OD matrix.
        params: Validated solver configuration.
        output: Queue receiving a single terminal result or failure.
        cancelled: Cooperative cancellation flag shared with the parent.
    """

    def check_cancelled(_event: dict[str, object]) -> None:
        """Check cancellation at solver checkpoints without retaining progress.

        Args:
            _event: Lightweight solver progress, discarded immediately.

        Raises:
            SearchCancelled: The parent requested cancellation.
        """
        if cancelled.is_set():
            raise SearchCancelled()

    try:
        matrix = ODMatrix.from_csv(matrix_path)
        if params["algorithm"] == "dijkstra":
            result = shortest_route(
                matrix,
                params["start"],
                params["end"],
                to_minutes(params["departure"]),
                params["budget_min"] * 60,
                progress=check_cancelled,
            )
            if result is None:
                raise ValueError("No feasible route found within the time budget")
        else:
            solver = {"alns": ALNS, "vns": VNS, "aco": AntColony}[params["algorithm"]](
                iterations=params["iterations"],
                seed=params["seed"],
                progress=check_cancelled,
            )
            result = solver(
                matrix,
                params["start"],
                params["end"],
                to_minutes(params["departure"]),
                params["budget_min"] * 60,
            )
        check_cancelled({})
        output.put(
            {"status": "result", "result": route_snapshot(result), "message": None}
        )
    except SearchCancelled:
        pass
    except Exception as exc:  # noqa: BLE001 -- process boundary must report failures
        if not cancelled.is_set():
            output.put({"status": "error", "result": None, "message": str(exc)})


class SearchRun:
    """Own one isolated computation and retain only its terminal outcome.

    Attributes:
        id: Unique public identifier for this run.
        params: Validated inputs used by the child process.
        status: Running or terminal state.
        result: Final route, present only after successful completion.
        message: Failure explanation, if computation failed.
        process: Spawned solver process, separate from the API event loop.
    """

    def __init__(self, matrix_path: str | Path, params: SearchParameters) -> None:
        """Start a solver and its terminal-result collection thread.

        Args:
            matrix_path: Matrix CSV used by this run.
            params: Validated search inputs to copy to the child process.
        """
        self.id = uuid.uuid4().hex
        self.params = params
        self.status: RunStatus = "running"
        self.result: RouteSnapshot | None = None
        self.message: str | None = None
        self.lock = threading.Lock()
        ctx = mp.get_context("spawn")
        self.output: ProcessQueue[RunStatusPayload] = ctx.Queue(maxsize=1)
        self.cancelled: ProcessEvent = ctx.Event()
        self.process: BaseProcess = ctx.Process(
            target=search_worker,
            args=(str(matrix_path), params, self.output, self.cancelled),
            daemon=True,
        )
        self.process.start()
        self.collector = threading.Thread(target=self._collect, daemon=True)
        self.collector.start()

    def _finish(self, outcome: RunStatusPayload) -> None:
        """Atomically retain a terminal outcome if the run is still active.

        Args:
            outcome: Completed result, failure, or cancellation record.
        """
        with self.lock:
            if self.status in TERMINAL:
                return
            self.status = outcome["status"]
            self.result = outcome["result"]
            self.message = outcome["message"]

    def _collect(self) -> None:
        """Receive the final outcome and report unexpected worker exits."""
        try:
            while True:
                try:
                    outcome = self.output.get(timeout=0.2)
                except queue.Empty:
                    if self.process.is_alive():
                        continue
                    outcome: RunStatusPayload = {
                        "status": "stopped" if self.cancelled.is_set() else "error",
                        "result": None,
                        "message": None
                        if self.cancelled.is_set()
                        else "Solver process exited unexpectedly",
                    }
                self._finish(outcome)
                break
        finally:
            self.process.join(timeout=1)
            self.output.close()

    def snapshot(self) -> RunStatusPayload:
        """Read the status and final result under the same lock.

        Returns:
            Computation status, final route if successful, and failure message.
        """
        with self.lock:
            return {
                "status": self.status,
                "result": self.result,
                "message": self.message,
            }

    def stop(self) -> None:
        """Cancel with bounded waits, preserving an already-completed result."""
        self.cancelled.set()
        self.process.join(timeout=0.4)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
        self._finish({"status": "stopped", "result": None, "message": None})
        self.collector.join(timeout=1)


def point_metadata(matrix_path: str | Path) -> list[PointMetadata]:
    """Join matrix-owned coordinates/scores with configured display names.

    Args:
        matrix_path: Selected OD matrix CSV.

    Returns:
        Location metadata sorted by point ID; unknown display names use IDs.

    Raises:
        OSError: Matrix or configuration files cannot be read.
        ValueError: The matrix or coordinate data is inconsistent.
        KeyError: Required input fields or configuration keys are absent.
    """
    coordinates = load_points(matrix_path)
    matrix = ODMatrix.from_csv(matrix_path)
    names: dict[str, str] = {}
    for filename, variable, key in (
        ("hubs.py", "HUBS", "hub_id"),
        ("pois.py", "POIS", "poi_id"),
    ):
        for point in runpy.run_path(str(ROOT / "configs" / filename))[variable]:
            point_id = point[key] if key == "hub_id" else f"P{int(point[key]):03d}"
            names[point_id] = point["name"]
    types: dict[str, str] = {}
    with open(matrix_path, encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            for prefix in ("from", "to"):
                types[row[f"{prefix}_id"]] = row.get(f"{prefix}_type", "point")
    return [
        {
            "id": key,
            "name": names.get(key, key),
            "lat": lat,
            "lon": lon,
            "kind": types.get(key, "point"),
            "score": matrix.scores.get(key, 0),
        }
        for key, (lat, lon) in sorted(coordinates.items())
    ]


def load_road_paths(points: list[PointMetadata]) -> RoadPaths:
    """Load the configured OSM graph and build its per-leg geometry cache.

    Args:
        points: Matrix locations to snap to graph nodes.

    Returns:
        A road-path resolver shared by all runs in this server session.

    Raises:
        OSError: Configuration or graph cache files cannot be read.
        RuntimeError: The existing OSM downloader exhausts its retry attempts.
        KeyError: Required configuration fields are absent.
    """
    import yaml

    from busroute.graph.osm_fetch import fetch_road_graph

    cfg = yaml.safe_load((ROOT / "configs/settings.yaml").read_text(encoding="utf-8"))
    osm = cfg["osm"]
    graph = fetch_road_graph(
        place=osm.get("place"),
        network_type=osm["network_type"],
        cache_dir=str(ROOT / osm["cache_dir"]),
        bbox=osm.get("bbox"),
    )
    return RoadPaths(
        graph,
        {p["id"]: (p["lat"], p["lon"]) for p in points},
        cfg["poi"]["max_snap_distance_m"],
    )


class AppState:
    """Own matrix metadata, one background graph load, and one current run.

    Attributes:
        matrix_path: Absolute path shared with solver workers.
        points: Public marker and endpoint metadata.
        ids: IDs accepted by run and geometry requests.
        graph_status: Loading, ready, or error state for road reconstruction.
        graph_error: Background initialization failure, if any.
        paths: Shared geometry resolver once initialization succeeds.
        run: Latest run, retained until replaced or the server exits.
    """

    def __init__(self, matrix_path: str | Path, graph_loader: GraphLoader) -> None:
        """Load lightweight metadata; defer the large graph to startup.

        Args:
            matrix_path: Selected matrix CSV path.
            graph_loader: Callable that creates a road resolver from metadata.
        """
        self.matrix_path = Path(matrix_path).resolve()
        self.points: list[PointMetadata] = point_metadata(self.matrix_path)
        self.ids: set[str] = {p["id"] for p in self.points}
        self.graph_loader: GraphLoader = graph_loader
        self.graph_status: GraphStatus = "loading"
        self.graph_error: str | None = None
        self.paths: RoadPaths | None = None
        self.geometry_lock = threading.Lock()
        self.run_lock = threading.Lock()
        self.run: SearchRun | None = None
        self.graph_loads = 0
        self.graph_seconds: float | None = None
        self.closed = False

    def initialize_graph(self) -> None:
        """Initialize road geometry once and retain a readable failure state."""
        started = time.perf_counter()
        self.graph_loads += 1
        try:
            self.paths = self.graph_loader(self.points)
            self.graph_seconds = round(time.perf_counter() - started, 3)
            self.graph_status = "ready"
            logger.info(
                "OSM ready in %.3fs; graph loads=%d",
                self.graph_seconds,
                self.graph_loads,
            )
        except Exception as exc:  # noqa: BLE001 -- surface background loader failures
            self.graph_error = str(exc)
            self.graph_status = "error"

    def start(self, params: SearchParameters) -> SearchRun:
        """Start one search without overlapping another active worker.

        Args:
            params: Validated route-search inputs.

        Returns:
            The newly spawned run, replacing the previous terminal run.

        Raises:
            HTTPException: HTTP 409 for an active run, or 503 during shutdown.
        """
        with self.run_lock:
            if self.closed:
                raise HTTPException(503, "Server is shutting down")
            if self.run is not None and self.run.status not in TERMINAL:
                raise HTTPException(
                    409, "Stop the active search before starting another"
                )
            self.run = SearchRun(self.matrix_path, params)
            return self.run

    def get_run(self, run_id: str) -> SearchRun:
        """Resolve the current run ID.

        Args:
            run_id: Identifier returned when starting the run.

        Returns:
            The matching current run.

        Raises:
            HTTPException: HTTP 404 if the run is unknown or was replaced.
        """
        if self.run is None or self.run.id != run_id:
            raise HTTPException(404, "Run is no longer available")
        return self.run


def create_app(
    matrix_path: str | Path = ROOT / "data/final/od_matrix.csv",
    *,
    graph_loader: GraphLoader = load_road_paths,
) -> FastAPI:
    """Build a local FastAPI app with isolated searches and static web assets.

    Args:
        matrix_path: Matrix whose points populate the browser form.
        graph_loader: Road resolver factory, replaceable for focused tests.

    Returns:
        An application with REST endpoints and a lifespan-managed worker.

    Raises:
        OSError: Required matrix or configuration files cannot be read.
        ValueError: Startup metadata is invalid.
    """
    state = AppState(matrix_path, graph_loader)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        """Start graph loading and clean up the current worker on shutdown.

        Args:
            _app: FastAPI instance supplied by the lifespan protocol.

        Yields:
            Control to the serving application after starting its loader.
        """
        threading.Thread(
            target=state.initialize_graph, daemon=True, name="osm-loader"
        ).start()
        yield
        state.closed = True
        if state.run is not None:
            await asyncio.to_thread(state.run.stop)

    app = FastAPI(title="Da Nang Route Lab", lifespan=lifespan)
    app.state.lab = state

    @app.get("/")
    def index() -> FileResponse:
        """Return the browser application shell.

        Returns:
            The locally bundled HTML document.
        """
        return FileResponse(ASSETS / "index.html")

    @app.get("/api/status")
    def status() -> GraphStatusPayload:
        """Report graph readiness independently of solver computation.

        Returns:
            Graph status, load timing, and cached-leg diagnostics.
        """
        return {
            "graph": state.graph_status,
            "message": state.graph_error,
            "graph_loads": state.graph_loads,
            "graph_seconds": state.graph_seconds,
            "cached_legs": len(state.paths.cache) if state.paths is not None else 0,
        }

    @app.get("/api/bootstrap")
    def bootstrap() -> BootstrapPayload:
        """Provide location metadata, boundary geometry, and default inputs.

        Returns:
            The complete lightweight browser initialization payload.

        Raises:
            HTTPException: HTTP 422 if fewer than two points are available.
        """
        from busroute.config.restrictions import RING_FILE

        ids = sorted(state.ids)
        if len(ids) < 2:
            raise HTTPException(422, "The matrix needs at least two distinct points")
        start = "H01" if "H01" in state.ids else ids[0]
        end = (
            "H02"
            if "H02" in state.ids and start != "H02"
            else next(p for p in ids if p != start)
        )
        return {
            "points": state.points,
            "matrix": state.matrix_path.name,
            "ring": cast(JSONObject, json.loads(RING_FILE.read_text(encoding="utf-8"))),
            "defaults": {
                "algorithm": "alns",
                "start": start,
                "end": end,
                "departure": "08:00",
                "budget_min": 240,
                "iterations": 100,
                "seed": 0,
            },
            "status": status(),
        }

    @app.post("/api/geometry")
    def geometry(body: GeometryRequest) -> GeometryPayload:
        """Resolve a bounded batch of directed road geometries off the API loop.

        Args:
            body: Point pairs to reconstruct or retrieve from cache.

        Returns:
            Available geometries and explicit missing-path records.

        Raises:
            HTTPException: HTTP 422 for unknown points or 503 before readiness.
        """
        if any(a not in state.ids or b not in state.ids for a, b in body.legs):
            raise HTTPException(422, "Geometry requested for an unknown point")
        paths = state.paths
        if state.graph_status != "ready" or paths is None:
            raise HTTPException(
                503, state.graph_error or "Road geometry is still loading"
            )
        results: list[GeometryLeg] = []
        with state.geometry_lock:
            for origin, destination in dict.fromkeys(body.legs):
                segments = paths.leg(origin, destination)
                results.append(
                    {
                        "from": origin,
                        "to": destination,
                        "geometry": {"type": "MultiLineString", "coordinates": segments}
                        if segments is not None
                        else None,
                        "message": None
                        if segments is not None
                        else "No directed OSM path, or a stop is too far from the road graph",
                    }
                )
        return {"legs": results}

    @app.post("/api/runs", status_code=201)
    def start_run(body: RunRequest) -> RunCreatedPayload:
        """Validate matrix membership and spawn a route-search worker.

        Args:
            body: Validated search parameters from the browser.

        Returns:
            The run ID used for status polling, cancellation, and result download.

        Raises:
            HTTPException: HTTP 422 for unknown points or 409 for overlap.
        """
        if body.start not in state.ids or body.end not in state.ids:
            raise HTTPException(
                422, "Start and destination must be in the selected matrix"
            )
        run = state.start(cast(SearchParameters, body.model_dump()))
        return {"id": run.id}

    @app.get("/api/runs/{run_id}")
    def run_status(run_id: str) -> RunStatusPayload:
        """Report computation status and the final route without search steps.

        Args:
            run_id: Identifier returned when starting the computation.

        Returns:
            Running or terminal status, final result, and failure message.

        Raises:
            HTTPException: HTTP 404 for an unknown or replaced run.
        """
        return state.get_run(run_id).snapshot()

    @app.post("/api/runs/{run_id}/stop")
    def stop_run(run_id: str) -> RunStoppedPayload:
        """Cancel the current process, preserving any already-completed result.

        Args:
            run_id: Identifier of the run to stop.

        Returns:
            Its terminal status, including already-completed success.

        Raises:
            HTTPException: HTTP 404 if the run is no longer available.
        """
        run = state.get_run(run_id)
        run.stop()
        return {"status": run.status}

    @app.get("/api/runs/{run_id}/result")
    def result(run_id: str) -> JSONResponse:
        """Download the final matrix-evaluated route as JSON.

        Args:
            run_id: Identifier of a successfully completed run.

        Returns:
            A JSON attachment named route.json.

        Raises:
            HTTPException: HTTP 404 for a missing run or 409 before success.
        """
        run = state.get_run(run_id)
        if run.status != "result":
            raise HTTPException(409, "A completed result is not available")
        return JSONResponse(
            run.result,
            headers={"Content-Disposition": 'attachment; filename="route.json"'},
        )

    app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")
    return app
