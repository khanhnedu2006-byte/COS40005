"""Đánh giá và so sánh ma trận [poi, street-reduced, street-full] × [dijkstra, alns, vns, aco, ppo]."""

from __future__ import annotations

import csv
import json
import math
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from busroute.graph.ab_estimator import PARAMS_FILE
from busroute.routing.base import RouteRequest
from busroute.routing.edge_metaheuristics import (
    STREET_SOLVERS,
    StreetContext,
)
from busroute.routing.ppo_env import REDUCED_DIR, GraphData
from busroute.routing.ppo_train import CHECKPOINT, CHECKPOINT_FULL, StreetPPOSolver
from busroute.routing.ppo_waypoint import CHECKPOINT_WAYPOINT, PoiPPOSolver
from busroute.routing.road_dijkstra import road_widths
from busroute.routing.road_graph import (
    RoadGraph,
    RoadProblem,
    configured_points,
    resolve_road_tables,
)
from busroute.routing.road_metaheuristics import POI_SOLVERS, PoiContext
from busroute.utils.timeslots import slot_of, to_minutes

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NODES = ROOT / "data" / "stage1" / "nodes_base.parquet"
DEFAULT_EDGES = ROOT / "data" / "stage1" / "edges_base.parquet"
DEFAULT_WIDTHS = ROOT / "data" / "final" / "duong_pho_clean.csv"
DEFAULT_REDUCED_DIR = REDUCED_DIR

DEFAULT_POI_CHECKPOINT = CHECKPOINT_WAYPOINT
DEFAULT_WAYPOINT_CHECKPOINT = CHECKPOINT_WAYPOINT
DEFAULT_REDUCED_CHECKPOINT = CHECKPOINT
DEFAULT_CHECKPOINT = CHECKPOINT
DEFAULT_FULL_CHECKPOINT = CHECKPOINT_FULL

ALL_ALGORITHMS: tuple[str, ...] = ("dijkstra", "alns", "vns", "aco", "ppo")
ALL_MODES: tuple[str, ...] = ("poi", "street-reduced", "street-full")
MODE_ALIASES: dict[str, str] = {
    "waypoint": "poi",
    "edge": "street-reduced",
}
METAHEURISTIC_ALGORITHMS: tuple[str, ...] = ("alns", "vns", "aco")


@dataclass
class AlgorithmEvaluation:
    """Kết quả đánh giá chi tiết của một cặp (thuật toán, chế độ)."""

    algorithm: str
    status: str
    mode: str = "poi"
    error: str | None = None
    duration_min: float | None = None
    poi_count: int | None = None
    poi_score: float | None = None
    distance_km: float | None = None
    budget_util_pct: float | None = None
    within_budget: bool | None = None
    score_per_min: float | None = None
    score_per_km: float | None = None
    narrow_m: float | None = None
    narrow_pct: float | None = None
    unknown_width_m: float | None = None
    avg_speed_kmh: float | None = None
    waypoint_count: int | None = None
    instruction_count: int | None = None
    solve_time_s: float | None = None
    detour_time_ratio: float | None = None
    detour_dist_ratio: float | None = None
    score_gain_vs_dijkstra: float | None = None
    route: list[str] = field(default_factory=list)
    collected: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Chuyển kết quả đánh giá sang dictionary tương thích JSON."""
        return asdict(self)

    def to_flat_row(self) -> dict[str, Any]:
        """Làm phẳng kết quả thành một dòng phục vụ xuất CSV."""
        row = self.to_dict()
        row["route"] = ";".join(self.route)
        row["collected"] = ";".join(self.collected)
        return row


def parse_departure(departure: str) -> int:
    """Kiểm tra định dạng giờ khởi hành ``HH:MM`` và đổi sang phút trong ngày."""
    if (
        not isinstance(departure, str)
        or not re.fullmatch(r"[0-2][0-9]:[0-5][0-9]", departure)
        or int(departure[:2]) >= 24
    ):
        raise ValueError("Departure must be a valid HH:MM time (00:00..23:59)")
    return to_minutes(departure)


def parse_algorithms(raw: str | Sequence[str]) -> tuple[str, ...]:
    """Chuẩn hóa danh sách thuật toán, giữ nguyên thứ tự và loại trùng lặp."""
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    selected: list[str] = []
    for item in items:
        name = item.strip().lower()
        if not name:
            continue
        if name not in ALL_ALGORITHMS:
            valid = ", ".join(ALL_ALGORITHMS)
            raise ValueError(f"Unknown algorithm '{name}'. Choose from: {valid}")
        if name not in selected:
            selected.append(name)
    if not selected:
        raise ValueError("At least one algorithm must be selected")
    return tuple(selected)


def parse_modes(raw: str | Sequence[str]) -> tuple[str, ...]:
    """Chuẩn hóa danh sách chế độ (``poi``, ``street-reduced``, ``street-full``, ``street``, ``all``)."""
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    selected: list[str] = []

    def add_mode(mode_name: str) -> None:
        if mode_name not in selected:
            selected.append(mode_name)

    for item in items:
        name = item.strip().lower()
        if not name:
            continue
        if name == "all":
            for m in ALL_MODES:
                add_mode(m)
            continue
        if name == "both":
            add_mode("poi")
            add_mode("street-reduced")
            continue
        if name == "street":
            add_mode("street-reduced")
            add_mode("street-full")
            continue
        canonical = MODE_ALIASES.get(name, name)
        if canonical not in ALL_MODES:
            valid = ", ".join((*ALL_MODES, "street", "all", "both", *MODE_ALIASES))
            raise ValueError(f"Unknown mode '{name}'. Choose from: {valid}")
        add_mode(canonical)
    if not selected:
        raise ValueError("At least one mode must be selected")
    return tuple(selected)


def metrics_from_snapshot(
    algorithm: str,
    snapshot: dict[str, Any],
    *,
    mode: str = "poi",
    budget_min: float,
    solve_time_s: float,
) -> AlgorithmEvaluation:
    """Tính toán toàn bộ chỉ số định lượng từ ``RouteSnapshot`` trả về bởi solver."""
    duration_min = float(snapshot["duration_min"])
    distance_km = float(snapshot["distance_km"])
    poi_score = float(snapshot["score"])
    collected = list(snapshot.get("collected", []))
    route = list(snapshot.get("route", []))
    narrow_m = float(snapshot.get("narrow_m", 0.0))
    unknown_width_m = float(snapshot.get("unknown_width_m", 0.0))
    instructions = snapshot.get("instructions", [])

    distance_m = distance_km * 1000.0
    budget_util_pct = (duration_min / budget_min * 100.0) if budget_min > 0 else None
    score_per_min = (poi_score / duration_min) if duration_min > 0 else None
    score_per_km = (poi_score / distance_km) if distance_km > 0 else None
    narrow_pct = (narrow_m / distance_m * 100.0) if distance_m > 0 else None
    avg_speed_kmh = (distance_km / (duration_min / 60.0)) if duration_min > 0 else None

    return AlgorithmEvaluation(
        algorithm=algorithm,
        mode=mode,
        status="ok",
        duration_min=duration_min,
        poi_count=len(collected),
        poi_score=poi_score,
        distance_km=distance_km,
        budget_util_pct=budget_util_pct,
        within_budget=duration_min <= budget_min + 1e-6,
        score_per_min=score_per_min,
        score_per_km=score_per_km,
        narrow_m=narrow_m,
        narrow_pct=narrow_pct,
        unknown_width_m=unknown_width_m,
        avg_speed_kmh=avg_speed_kmh,
        waypoint_count=len(route),
        instruction_count=len(instructions),
        solve_time_s=solve_time_s,
        route=route,
        collected=collected,
    )


def _load_graph_data(
    nodes_path: Path,
    edges_path: Path,
    widths_path: Path,
    reduced_dir: Path,
    points: list[dict[str, Any]],
    *,
    use_reduced: bool,
) -> tuple[GraphData, Any]:
    """Nạp ``GraphData`` và ``orig_coords`` dùng chung cho chế độ ``street-*``."""
    nodes_df, edges_df, orig_coords = resolve_road_tables(
        nodes_path, edges_path, reduced_dir=reduced_dir, use_reduced=use_reduced
    )
    widths = road_widths(widths_path)
    params = json.loads(Path(PARAMS_FILE).read_text(encoding="utf-8"))
    graph = GraphData(nodes_df, edges_df, points, params, widths=widths)
    return graph, orig_coords


def _apply_dijkstra_baseline(results: list[AlgorithmEvaluation]) -> None:
    """Bổ sung tỉ lệ đi vòng và chênh lệch điểm POI so với baseline Dijkstra cùng chế độ."""
    dijkstra_row = next(
        (row for row in results if row.algorithm == "dijkstra" and row.status == "ok"),
        None,
    )
    if dijkstra_row is None:
        return

    base_dur = dijkstra_row.duration_min
    base_dist = dijkstra_row.distance_km
    base_score = dijkstra_row.poi_score
    for row in results:
        if row.status != "ok":
            continue
        if base_dur is not None and base_dur > 0 and row.duration_min is not None:
            row.detour_time_ratio = row.duration_min / base_dur
        if base_dist is not None and base_dist > 0 and row.distance_km is not None:
            row.detour_dist_ratio = row.distance_km / base_dist
        if base_score is not None and row.poi_score is not None:
            row.score_gain_vs_dijkstra = row.poi_score - base_score


def _build_summary(results: list[AlgorithmEvaluation]) -> dict[str, dict[str, Any] | None]:
    """Tìm cặp (chế độ, thuật toán) tốt nhất theo từng tiêu chí trong số các lượt thành công."""
    ok = [row for row in results if row.status == "ok"]
    if not ok:
        return {
            "best_poi_score": None,
            "most_pois": None,
            "fastest_route": None,
            "shortest_distance": None,
            "lowest_narrow_m": None,
            "fastest_solve_time": None,
        }

    def pick_best(attr: str, *, maximize: bool) -> dict[str, Any] | None:
        candidates = [row for row in ok if getattr(row, attr) is not None]
        if not candidates:
            return None
        winner = (max if maximize else min)(
            candidates, key=lambda row: getattr(row, attr)
        )
        return {
            "algorithm": winner.algorithm,
            "mode": winner.mode,
            "value": getattr(winner, attr),
        }

    return {
        "best_poi_score": pick_best("poi_score", maximize=True),
        "most_pois": pick_best("poi_count", maximize=True),
        "fastest_route": pick_best("duration_min", maximize=False),
        "shortest_distance": pick_best("distance_km", maximize=False),
        "lowest_narrow_m": pick_best("narrow_m", maximize=False),
        "fastest_solve_time": pick_best("solve_time_s", maximize=False),
    }


def _run_poi_mode(
    alg_list: tuple[str, ...],
    start: str,
    end: str,
    budget_min: float,
    start_minute: int,
    iterations: int,
    seed: int,
    workers: int,
    poi_checkpoint: str | Path | None,
    nodes_p: Path,
    edges_p: Path,
    widths_p: Path,
    reduced_p: Path,
    points_list: list[dict[str, Any]],
    timing: dict[str, float | None],
) -> list[AlgorithmEvaluation]:
    """Chạy tập thuật toán ở chế độ ``poi`` trên ``RoadGraph`` / ``RoadProblem``."""
    t0 = time.perf_counter()
    road_graph = RoadGraph.from_files(
        nodes_p, edges_p, widths_p, points_list, reduced_dir=reduced_p, use_reduced=True
    )
    timing["poi_graph_load_s"] = round(time.perf_counter() - t0, 4)

    request = RouteRequest(
        start=start,
        end=end,
        budget_min=budget_min,
        start_minute=start_minute,
        iterations=iterations,
        seed=seed,
        workers=workers,
    )
    by_alg: dict[str, AlgorithmEvaluation] = {}
    if "dijkstra" in alg_list:
        t_solve = time.perf_counter()
        try:
            dijk_problem = RoadProblem(
                road_graph,
                start,
                end,
                budget_min,
                start_minute=start_minute,
                all_pois=False,
                workers=workers,
            )
            snapshot = POI_SOLVERS["dijkstra"]().solve_snapshot(
                request,
                context=PoiContext(road_graph=road_graph, problem=dijk_problem),
            )
            by_alg["dijkstra"] = metrics_from_snapshot(
                "dijkstra",
                snapshot,
                mode="poi",
                budget_min=budget_min,
                solve_time_s=time.perf_counter() - t_solve,
            )
        except Exception as exc:  # noqa: BLE001
            by_alg["dijkstra"] = AlgorithmEvaluation(
                algorithm="dijkstra",
                mode="poi",
                status="error",
                error=str(exc),
                solve_time_s=time.perf_counter() - t_solve,
            )

    need_full_problem = [a for a in alg_list if a in (*METAHEURISTIC_ALGORITHMS, "ppo")]
    if need_full_problem:
        t_prep = time.perf_counter()
        try:
            meta_problem = RoadProblem(
                road_graph,
                start,
                end,
                budget_min,
                start_minute=start_minute,
                all_pois=True,
                workers=workers,
            )
            meta_problem.trees(slot_of(start_minute))
            timing["poi_prep_s"] = round(time.perf_counter() - t_prep, 4)
        except Exception as exc:  # noqa: BLE001
            timing["poi_prep_s"] = round(time.perf_counter() - t_prep, 4)
            for alg in need_full_problem:
                by_alg[alg] = AlgorithmEvaluation(
                    algorithm=alg,
                    mode="poi",
                    status="error",
                    error=str(exc),
                    solve_time_s=0.0,
                )
        else:
            poi_ctx = PoiContext(road_graph=road_graph, problem=meta_problem)
            # Chạy các metaheuristic (có thể fork pool) trước khi nạp PyTorch cho PPO.
            for alg in [a for a in alg_list if a in METAHEURISTIC_ALGORITHMS]:
                t_solve = time.perf_counter()
                try:
                    snapshot = POI_SOLVERS[alg]().solve_snapshot(request, context=poi_ctx)
                    by_alg[alg] = metrics_from_snapshot(
                        alg,
                        snapshot,
                        mode="poi",
                        budget_min=budget_min,
                        solve_time_s=time.perf_counter() - t_solve,
                    )
                except Exception as exc:  # noqa: BLE001
                    by_alg[alg] = AlgorithmEvaluation(
                        algorithm=alg,
                        mode="poi",
                        status="error",
                        error=str(exc),
                        solve_time_s=time.perf_counter() - t_solve,
                    )

            if "ppo" in alg_list:
                ckpt_path = Path(poi_checkpoint) if poi_checkpoint is not None else None
                if ckpt_path is None:
                    by_alg["ppo"] = AlgorithmEvaluation(
                        algorithm="ppo",
                        mode="poi",
                        status="error",
                        error="No POI PPO checkpoint provided",
                        solve_time_s=0.0,
                    )
                elif not ckpt_path.is_file():
                    by_alg["ppo"] = AlgorithmEvaluation(
                        algorithm="ppo",
                        mode="poi",
                        status="error",
                        error=f"POI PPO checkpoint not found: {ckpt_path}",
                        solve_time_s=0.0,
                    )
                else:
                    t_load = time.perf_counter()
                    try:
                        from sb3_contrib import MaskablePPO

                        wp_model = MaskablePPO.load(ckpt_path, device="cpu")
                        timing["poi_ppo_load_s"] = round(time.perf_counter() - t_load, 4)
                    except Exception as exc:  # noqa: BLE001
                        timing["poi_ppo_load_s"] = round(time.perf_counter() - t_load, 4)
                        by_alg["ppo"] = AlgorithmEvaluation(
                            algorithm="ppo",
                            mode="poi",
                            status="error",
                            error=str(exc),
                            solve_time_s=0.0,
                        )
                    else:
                        t_solve = time.perf_counter()
                        try:
                            snapshot = PoiPPOSolver(
                                ckpt_path, model=wp_model
                            ).solve_snapshot(request, context=poi_ctx)
                            by_alg["ppo"] = metrics_from_snapshot(
                                "ppo",
                                snapshot,
                                mode="poi",
                                budget_min=budget_min,
                                solve_time_s=time.perf_counter() - t_solve,
                            )
                        except Exception as exc:  # noqa: BLE001
                            by_alg["ppo"] = AlgorithmEvaluation(
                                algorithm="ppo",
                                mode="poi",
                                status="error",
                                error=str(exc),
                                solve_time_s=time.perf_counter() - t_solve,
                            )

    rows = [by_alg[alg] for alg in alg_list]
    _apply_dijkstra_baseline(rows)
    return rows


def _run_street_mode(
    alg_list: tuple[str, ...],
    start: str,
    end: str,
    budget_min: float,
    departure: str,
    start_minute: int,
    iterations: int,
    seed: int,
    workers: int,
    safety_margin: float,
    mode: str,
    use_reduced: bool,
    edge_checkpoint: str | Path | None,
    nodes_p: Path,
    edges_p: Path,
    widths_p: Path,
    reduced_p: Path,
    points_list: list[dict[str, Any]],
    timing: dict[str, float | None],
) -> list[AlgorithmEvaluation]:
    """Chạy tập thuật toán ở chế độ ``street-reduced`` hoặc ``street-full`` trên ``GraphData``."""
    timing_prefix = "street_reduced" if use_reduced else "street_full"
    t0 = time.perf_counter()
    street_graph, orig_coords = _load_graph_data(
        nodes_p,
        edges_p,
        widths_p,
        reduced_p,
        points_list,
        use_reduced=use_reduced,
    )
    timing[f"{timing_prefix}_graph_load_s"] = round(time.perf_counter() - t0, 4)

    request = RouteRequest(
        start=start,
        end=end,
        budget_min=budget_min,
        start_minute=start_minute,
        iterations=iterations,
        seed=seed,
        workers=workers,
        safety_margin=safety_margin,
    )
    street_ctx = StreetContext(graph=street_graph, orig_coords=orig_coords)

    by_alg: dict[str, AlgorithmEvaluation] = {}
    # Chạy Dijkstra và các metaheuristic (có thể fork pool) trước khi nạp PyTorch cho PPO.
    for alg in [a for a in alg_list if a != "ppo"]:
        t_solve = time.perf_counter()
        try:
            snapshot = STREET_SOLVERS[alg]().solve_snapshot(request, context=street_ctx)
            by_alg[alg] = metrics_from_snapshot(
                alg,
                snapshot,
                mode=mode,
                budget_min=budget_min,
                solve_time_s=time.perf_counter() - t_solve,
            )
        except Exception as exc:  # noqa: BLE001
            by_alg[alg] = AlgorithmEvaluation(
                algorithm=alg,
                mode=mode,
                status="error",
                error=str(exc),
                solve_time_s=time.perf_counter() - t_solve,
            )

    if "ppo" in alg_list:
        ckpt_path = Path(edge_checkpoint) if edge_checkpoint is not None else None
        if ckpt_path is None:
            by_alg["ppo"] = AlgorithmEvaluation(
                algorithm="ppo",
                mode=mode,
                status="error",
                error=f"No {mode} PPO checkpoint provided",
                solve_time_s=0.0,
            )
        elif not ckpt_path.is_file():
            by_alg["ppo"] = AlgorithmEvaluation(
                algorithm="ppo",
                mode=mode,
                status="error",
                error=f"{mode} PPO checkpoint not found: {ckpt_path}",
                solve_time_s=0.0,
            )
        else:
            t_load = time.perf_counter()
            try:
                from sb3_contrib import MaskablePPO

                edge_model = MaskablePPO.load(ckpt_path, device="cpu")
                timing[f"{timing_prefix}_ppo_load_s"] = round(
                    time.perf_counter() - t_load, 4
                )
            except Exception as exc:  # noqa: BLE001
                timing[f"{timing_prefix}_ppo_load_s"] = round(
                    time.perf_counter() - t_load, 4
                )
                by_alg["ppo"] = AlgorithmEvaluation(
                    algorithm="ppo",
                    mode=mode,
                    status="error",
                    error=str(exc),
                    solve_time_s=0.0,
                )
            else:
                t_solve = time.perf_counter()
                try:
                    snapshot = StreetPPOSolver(
                        ckpt_path, model=edge_model
                    ).solve_snapshot(request, context=street_ctx)
                    by_alg["ppo"] = metrics_from_snapshot(
                        "ppo",
                        snapshot,
                        mode=mode,
                        budget_min=budget_min,
                        solve_time_s=time.perf_counter() - t_solve,
                    )
                except Exception as exc:  # noqa: BLE001
                    by_alg["ppo"] = AlgorithmEvaluation(
                        algorithm="ppo",
                        mode=mode,
                        status="error",
                        error=str(exc),
                        solve_time_s=time.perf_counter() - t_solve,
                    )

    rows = [by_alg[alg] for alg in alg_list]
    _apply_dijkstra_baseline(rows)
    return rows


def compare_algorithms(
    start: str,
    end: str,
    budget_min: float,
    *,
    departure: str = "08:00",
    algorithms: Sequence[str] = ALL_ALGORITHMS,
    modes: str | Sequence[str] = ("poi",),
    checkpoint: str | Path | None = DEFAULT_REDUCED_CHECKPOINT,
    full_checkpoint: str | Path | None = DEFAULT_FULL_CHECKPOINT,
    poi_checkpoint: str | Path | None = None,
    waypoint_checkpoint: str | Path | None = None,
    iterations: int = 100,
    seed: int = 0,
    workers: int = 1,
    safety_margin: float = 1.05,
    nodes_path: str | Path = DEFAULT_NODES,
    edges_path: str | Path = DEFAULT_EDGES,
    widths_path: str | Path = DEFAULT_WIDTHS,
    reduced_dir: str | Path = DEFAULT_REDUCED_DIR,
    points: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Chạy và so sánh ma trận ``modes × algorithms`` trên cùng điểm đầu/cuối và ngân sách."""
    alg_list = parse_algorithms(algorithms)
    mode_list = parse_modes(modes)
    start_minute = parse_departure(departure)
    if not math.isfinite(budget_min) or budget_min <= 0:
        raise ValueError("Time budget must be positive and finite")
    if isinstance(iterations, bool) or iterations < 0:
        raise ValueError("iterations must be a non-negative integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not math.isfinite(safety_margin) or safety_margin < 1.0:
        raise ValueError("safety_margin must be finite and >= 1.0")

    points_list = list(points) if points is not None else configured_points()
    point_map = {p["id"]: p for p in points_list}
    if start not in point_map:
        raise ValueError(f"Unknown start point '{start}'")
    if end not in point_map:
        raise ValueError(f"Unknown end point '{end}'")
    if start == end:
        raise ValueError("Start and end points must be distinct")

    resolved_poi_ckpt = (
        poi_checkpoint
        if poi_checkpoint is not None
        else (waypoint_checkpoint if waypoint_checkpoint is not None else DEFAULT_POI_CHECKPOINT)
    )

    nodes_p = Path(nodes_path)
    edges_p = Path(edges_path)
    widths_p = Path(widths_path)
    reduced_p = Path(reduced_dir)

    timing: dict[str, float | None] = {
        "poi_graph_load_s": None,
        "poi_prep_s": None,
        "poi_ppo_load_s": None,
        "street_reduced_graph_load_s": None,
        "street_reduced_ppo_load_s": None,
        "street_full_graph_load_s": None,
        "street_full_ppo_load_s": None,
    }
    ordered_results: list[AlgorithmEvaluation] = []

    for mode in mode_list:
        if mode == "poi":
            ordered_results.extend(
                _run_poi_mode(
                    alg_list,
                    start,
                    end,
                    budget_min,
                    start_minute,
                    iterations,
                    seed,
                    workers,
                    resolved_poi_ckpt,
                    nodes_p,
                    edges_p,
                    widths_p,
                    reduced_p,
                    points_list,
                    timing,
                )
            )
        elif mode == "street-reduced":
            ordered_results.extend(
                _run_street_mode(
                    alg_list,
                    start,
                    end,
                    budget_min,
                    departure,
                    start_minute,
                    iterations,
                    seed,
                    workers,
                    safety_margin,
                    "street-reduced",
                    True,
                    checkpoint,
                    nodes_p,
                    edges_p,
                    widths_p,
                    reduced_p,
                    points_list,
                    timing,
                )
            )
        else:
            ordered_results.extend(
                _run_street_mode(
                    alg_list,
                    start,
                    end,
                    budget_min,
                    departure,
                    start_minute,
                    iterations,
                    seed,
                    workers,
                    safety_margin,
                    "street-full",
                    False,
                    full_checkpoint,
                    nodes_p,
                    edges_p,
                    widths_p,
                    reduced_p,
                    points_list,
                    timing,
                )
            )

    summary = _build_summary(ordered_results)
    return {
        "config": {
            "start": start,
            "start_name": point_map[start]["name"],
            "end": end,
            "end_name": point_map[end]["name"],
            "departure": departure,
            "departure_slot": slot_of(start_minute),
            "budget_min": float(budget_min),
            "iterations": iterations,
            "seed": seed,
            "workers": workers,
            "safety_margin": float(safety_margin),
            "modes": list(mode_list),
            "algorithms": list(alg_list),
            "poi_checkpoint": (
                str(resolved_poi_ckpt)
                if "poi" in mode_list and "ppo" in alg_list and resolved_poi_ckpt is not None
                else None
            ),
            "reduced_checkpoint": (
                str(checkpoint)
                if "street-reduced" in mode_list and "ppo" in alg_list and checkpoint is not None
                else None
            ),
            "full_checkpoint": (
                str(full_checkpoint)
                if "street-full" in mode_list and "ppo" in alg_list and full_checkpoint is not None
                else None
            ),
        },
        "timing": timing,
        "results": [row.to_dict() for row in ordered_results],
        "summary": summary,
    }


def _fmt(value: float | int | None, fmt: str, fallback: str = "-") -> str:
    if value is None:
        return fallback
    return format(value, fmt)


def format_report(report: dict[str, Any], *, show_pois: bool = False) -> str:
    """Định dạng báo cáo so sánh ma trận chế độ × thuật toán thành bảng text."""
    cfg = report["config"]
    timing = report["timing"]
    results = report["results"]
    summary = report["summary"]
    modes = cfg.get("modes") or ["poi"]

    lines = [
        "=" * 130,
        f"Route Evaluation: {cfg['start']} ({cfg['start_name']}) -> {cfg['end']} ({cfg['end_name']})",
        (
            f"Departure: {cfg['departure']} [{cfg['departure_slot']}] | "
            f"Budget: {cfg['budget_min']:.1f} min | Modes: {', '.join(modes)} | "
            f"Iterations: {cfg['iterations']} | Seed: {cfg['seed']} | "
            f"Workers: {cfg.get('workers', 1)} | Margin: {cfg.get('safety_margin', 1.05):.2f}"
        ),
    ]
    if cfg.get("poi_checkpoint"):
        lines.append(f"PPO [poi] checkpoint:            {cfg['poi_checkpoint']}")
    if cfg.get("reduced_checkpoint"):
        lines.append(f"PPO [street-reduced] checkpoint: {cfg['reduced_checkpoint']}")
    if cfg.get("full_checkpoint"):
        lines.append(f"PPO [street-full] checkpoint:    {cfg['full_checkpoint']}")

    timing_parts = [
        f"{label}={value:.2f}s"
        for label, value in (
            ("poi_graph", timing.get("poi_graph_load_s")),
            ("poi_prep", timing.get("poi_prep_s")),
            ("poi_ppo_load", timing.get("poi_ppo_load_s")),
            ("reduced_graph", timing.get("street_reduced_graph_load_s")),
            ("reduced_ppo_load", timing.get("street_reduced_ppo_load_s")),
            ("full_graph", timing.get("street_full_graph_load_s")),
            ("full_ppo_load", timing.get("street_full_ppo_load_s")),
        )
        if value is not None
    ]
    if timing_parts:
        lines.append("Setup timings: " + ", ".join(timing_parts))
    lines.append("-" * 130)

    headers = [
        "Mode",
        "Algorithm",
        "Status",
        "Time(m)",
        "Budget%",
        "POIs",
        "Score",
        "Dist(km)",
        "Score/m",
        "Score/km",
        "Narrow(m)",
        "UnkW(m)",
        "Speed(km/h)",
        "Solve(s)",
    ]
    rows: list[list[str]] = []
    for row in results:
        mode = row.get("mode", "poi")
        if row["status"] != "ok":
            rows.append(
                [
                    mode,
                    row["algorithm"],
                    "ERR",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    _fmt(row.get("solve_time_s"), ".2f"),
                ]
            )
            continue
        rows.append(
            [
                mode,
                row["algorithm"],
                "OK",
                _fmt(row["duration_min"], ".2f"),
                _fmt(row["budget_util_pct"], ".1f"),
                _fmt(row["poi_count"], "d"),
                _fmt(row["poi_score"], ".2f"),
                _fmt(row["distance_km"], ".2f"),
                _fmt(row["score_per_min"], ".3f"),
                _fmt(row["score_per_km"], ".3f"),
                _fmt(row["narrow_m"], ".0f"),
                _fmt(row["unknown_width_m"], ".0f"),
                _fmt(row["avg_speed_kmh"], ".1f"),
                _fmt(row["solve_time_s"], ".2f"),
            ]
        )

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))
    ]
    align = ["<", "<", "<", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">"]

    def render_row(values: list[str]) -> str:
        return "  ".join(
            f"{val:{align[idx]}{widths[idx]}}" for idx, val in enumerate(values)
        )

    lines.append(render_row(headers))
    lines.append("  ".join("-" * w for w in widths))
    prev_mode: str | None = None
    for r in rows:
        if prev_mode is not None and r[0] != prev_mode:
            lines.append("  ".join("·" * w for w in widths))
        lines.append(render_row(r))
        prev_mode = r[0]

    errors = [row for row in results if row["status"] != "ok"]
    if errors:
        lines.append("-" * 130)
        lines.append("Errors:")
        for row in errors:
            lines.append(f"  - [{row.get('mode', 'poi')}] {row['algorithm']}: {row['error']}")

    highlight_specs = [
        ("Best POI score", "best_poi_score", ".2f", ""),
        ("Most POIs", "most_pois", "d", " POIs"),
        ("Fastest route", "fastest_route", ".2f", " min"),
        ("Shortest distance", "shortest_distance", ".2f", " km"),
        ("Lowest narrow road", "lowest_narrow_m", ".0f", " m"),
        ("Fastest solve", "fastest_solve_time", ".2f", " s"),
    ]
    highlight_lines = []
    multi_mode = len(modes) > 1
    for label, key, fmt, unit in highlight_specs:
        entry = summary.get(key)
        if entry is not None:
            who = (
                f"{entry['mode']}:{entry['algorithm']}"
                if multi_mode and entry.get("mode")
                else entry["algorithm"]
            )
            highlight_lines.append(
                f"  - {label}: {who} ({format(entry['value'], fmt)}{unit})"
            )
    if highlight_lines:
        lines.append("-" * 130)
        lines.append("Highlights:")
        lines.extend(highlight_lines)

    lines.append("-" * 130)
    lines.append(
        "Note: all modes credit POIs snapped to visited nodes (matching find_route); "
        "'poi' plans over precomputed POI legs on the reduced graph, while 'street-reduced' "
        "(44.6k nodes) and 'street-full' (93.8k nodes) navigate street intersections."
    )

    if show_pois:
        ok_rows = [row for row in results if row["status"] == "ok"]
        if ok_rows:
            lines.append("-" * 130)
            lines.append("Routes & Collected POIs:")
            for row in ok_rows:
                route_str = " -> ".join(row["route"])
                pois_str = ", ".join(row["collected"]) if row["collected"] else "(none)"
                lines.append(
                    f"  [{row.get('mode', 'poi')}:{row['algorithm']}] "
                    f"waypoints ({row['waypoint_count']}): {route_str}"
                )
                lines.append(f"    collected ({row['poi_count']}): {pois_str}")

    lines.append("=" * 130)
    return "\n".join(lines)


def save_json(report: dict[str, Any], path: str | Path) -> Path:
    """Lưu toàn bộ báo cáo so sánh ra tệp JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def save_csv(report: dict[str, Any], path: str | Path) -> Path:
    """Lưu bảng kết quả từng thuật toán ra tệp CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for row in report["results"]:
        item = dict(row)
        item["route"] = ";".join(item.get("route") or [])
        item["collected"] = ";".join(item.get("collected") or [])
        flat_rows.append(item)
    fieldnames = list(AlgorithmEvaluation(algorithm="", status="").to_flat_row().keys())
    with out.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    return out
