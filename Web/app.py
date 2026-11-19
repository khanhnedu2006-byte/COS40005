"""Local web API, isolated solver processes, and reusable OSM geometry."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import multiprocessing as mp
import queue
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
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from Web.road_paths import RoadPaths
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
    """Time-budgeted road search, with optional seeded heuristic iterations."""

    model_config = ConfigDict(extra="forbid")
    algorithm: AlgorithmName = "alns"
    start: str
    end: str
    budget_min: float = Field(default=240, gt=0, allow_inf_nan=False)
    departure: str = Field(default="08:00", pattern=r"^\d{2}:\d{2}$")
    iterations: int = Field(default=100, ge=0, strict=True)
    seed: int = Field(default=0, strict=True)
    checkpoint: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_route(self) -> RunRequest:
        if self.start == self.end:
            raise ValueError("Choose different start and destination points")
        hour, minute = map(int, self.departure.split(":"))
        if hour >= 24 or minute >= 60:
            raise ValueError("Departure must be a valid HH:MM time")
        if self.algorithm in ("ppo", "ppo_waypoint") and not self.checkpoint:
            raise ValueError("Choose a trained checkpoint for the selected PPO solver")
        return self


class GeometryRequest(BaseModel):
    """Request at most 200 directed point-pair geometries.

    Attributes:
        legs: Origin/destination ID pairs, deduplicated by the endpoint.
    """

    legs: list[tuple[str, str]] = Field(max_length=200)


def search_worker(road_files, points, params, output, cancelled) -> None:
    """Compute a final graph-derived route in an isolated process."""
    try:
        from busroute.routing.road_dijkstra import find_route

        result = find_route(
            *road_files,
            points,
            params["start"],
            params["end"],
            cancelled.is_set,
            algorithm=params["algorithm"],
            budget_min=params["budget_min"],
            departure=params["departure"],
            iterations=params["iterations"],
            seed=params["seed"],
            checkpoint=params.get("checkpoint"),
        )
        if not cancelled.is_set():
            output.put({"status": "result", "result": result, "message": None})
    except InterruptedError:
        pass
    except Exception as exc:
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

    def __init__(self, road_files, points, params: SearchParameters) -> None:
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
            args=(
                tuple(map(str, road_files)),
                points,
                params,
                self.output,
                self.cancelled,
            ),
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
    """Read endpoint and POI metadata from project configuration."""
    from busroute.routing.road_graph import configured_points

    return configured_points(ROOT)


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

    def __init__(self, matrix_path, graph_loader, road_files, points=None) -> None:
        """Load lightweight metadata; defer the large graph to startup.

        Args:
            matrix_path: Selected matrix CSV path.
            graph_loader: Callable that creates a road resolver from metadata.
        """
        self.matrix_path = Path(matrix_path).resolve()
        self.points: list[PointMetadata] = (
            points if points is not None else point_metadata(self.matrix_path)
        )
        self.road_files = tuple(Path(path).resolve() for path in road_files)
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
            if self.graph_status != "ready":
                raise HTTPException(503, self.graph_error or "Road data unavailable")
            if self.run is not None and self.run.status not in TERMINAL:
                raise HTTPException(
                    409, "Stop the active search before starting another"
                )
            self.run = SearchRun(self.road_files, self.points, params)
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
    nodes_path: str | Path = ROOT / "data/stage1/nodes_base.parquet",
    edges_path: str | Path = ROOT / "data/stage1/edges_base.parquet",
    widths_path: str | Path = ROOT / "data/final/duong_pho_clean.csv",
    points: list[PointMetadata] | None = None,
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
    state = AppState(
        matrix_path, graph_loader, (nodes_path, edges_path, widths_path), points
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        """Start graph loading and clean up the current worker on shutdown.

        Args:
            _app: FastAPI instance supplied by the lifespan protocol.

        Yields:
            Control to the serving application after starting its loader.
        """
        started = time.perf_counter()
        try:
            import pandas as pd
            from busroute.routing.road_graph import snap_points

            missing = [path.name for path in state.road_files if not path.is_file()]
            if missing:
                raise ValueError(f"Missing road files: {', '.join(missing)}")
            snapped, distances, _ = snap_points(
                pd.read_parquet(state.road_files[0]), state.points
            )
            for point in state.points:
                point["routing_available"] = point["id"] in snapped
                point["snap_distance_m"] = distances[point["id"]]
            state.graph_status = "ready"
            state.graph_loads = 1
        except Exception as exc:
            state.graph_status, state.graph_error = "error", str(exc)
        state.graph_seconds = round(time.perf_counter() - started, 3)
        yield
        state.closed = True
        if state.run is not None:
            await asyncio.to_thread(state.run.stop)

    app = FastAPI(title="Da Nang Route Lab", lifespan=lifespan)
    app.state.lab = state

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        """Return fresh HTML with content-versioned application assets.

        Returns:
            HTML whose script and stylesheet URLs change with their contents,
            preventing cached assets from targeting an outdated page structure.
        """
        html = (ASSETS / "index.html").read_text(encoding="utf-8")
        for filename in ("app.js", "app.css"):
            version = hashlib.sha256((ASSETS / filename).read_bytes()).hexdigest()[:12]
            html = html.replace(
                f'"/assets/{filename}"', f'"/assets/{filename}?v={version}"'
            )
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

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

        ids = sorted(p["id"] for p in state.points if p.get("routing_available", True))
        if len(ids) < 2:
            raise HTTPException(
                422, "At least two available road endpoints are required"
            )
        start = "H01" if "H01" in ids else ids[0]
        end = (
            "H02"
            if "H02" in ids and start != "H02"
            else next(p for p in ids if p != start)
        )
        return {
            "points": state.points,
            "matrix": "Road graph + road widths",
            "ring": cast(JSONObject, json.loads(RING_FILE.read_text(encoding="utf-8"))),
            "defaults": {
                "algorithm": "alns",
                "start": start,
                "end": end,
                "budget_min": 240,
                "departure": "08:00",
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
                422, "Start and destination must be configured road points"
            )
        unavailable = {
            p["id"] for p in state.points if not p.get("routing_available", True)
        }
        if body.start in unavailable or body.end in unavailable:
            raise HTTPException(
                422, "An endpoint is more than 300 m from the road graph"
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
