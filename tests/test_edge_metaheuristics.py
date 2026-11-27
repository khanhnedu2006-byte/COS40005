"""Kiểm thử các thuật toán định tuyến ở mức cạnh (Dijkstra, ALNS, VNS, ACO) trên GraphData."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from busroute.graph.ab_estimator import PARAMS_FILE
from busroute.routing.base import RoadDatasetSpec, RouteOutput, RouteRequest
from busroute.routing.edge_metaheuristics import (
    StreetACOSolver,
    StreetALNSSolver,
    StreetDijkstraSolver,
    _prune_zero_poi_cycles,
    _walk_ppo_return,
    edge_aco,
    edge_alns,
    edge_dijkstra,
    edge_vns,
    evaluate_edge_walk,
    find_edge_route,
    walk_to_snapshot,
)
from busroute.routing.ppo_env import BusRouteEnv, GraphData
from busroute.routing.road_dijkstra import road_widths
from busroute.routing.road_graph import NoRoadRouteError
from busroute.routing.road_metaheuristics import PoiACOSolver, PoiALNSSolver
from busroute.utils.timeslots import TIME_SLOTS, slot_of, to_minutes


@pytest.fixture
def edge_env(tmp_path: Path) -> dict:
    """Tạo đồ thị thử nghiệm có nhánh rẽ qua POI và đường song song có vòng lặp."""
    nodes = pd.DataFrame(
        [
            (0, 16.000, 108.000),
            (1, 16.000, 108.006),
            (2, 16.003, 108.003),
            (3, 16.006, 108.003),
        ],
        columns=["node_idx", "lat", "lon"],
    )
    edges = pd.DataFrame(
        [
            (0, 1, "Direct", 600.0, "0"),
            (1, 0, "Return", 600.0, ""),
            (0, 2, "To P", 450.0, ""),
            (2, 1, "From P", 450.0, ""),
            (2, 3, "To Q", 330.0, ""),
            (3, 2, "Back Q", 330.0, ""),
            (3, 1, "From Q", 670.0, ""),
            (0, 3, "To Q Direct", 900.0, ""),
        ],
        columns=["from_idx", "to_idx", "road_name", "length_m", "via_nodes"],
    )
    nodes_path = tmp_path / "nodes.parquet"
    edges_path = tmp_path / "edges.parquet"
    widths_path = tmp_path / "widths.csv"
    nodes.to_parquet(nodes_path)
    edges.to_parquet(edges_path)
    widths_path.write_text(
        "Tên đường,Chiều rộng mặt (m)\n"
        "Direct,4\n"
        "Return,8\n"
        "To P,8\n"
        "From P,8\n"
        "To Q,8\n"
        "Back Q,8\n"
        "From Q,8\n"
        "To Q Direct,8\n",
        encoding="utf-8-sig",
    )
    points = [
        {"id": "H01", "name": "Start", "kind": "hub", "score": 0.0, "lat": 16.000, "lon": 108.000},
        {"id": "H02", "name": "End", "kind": "hub", "score": 0.0, "lat": 16.000, "lon": 108.006},
        {"id": "P001", "name": "P1", "kind": "poi", "score": 5.0, "lat": 16.003, "lon": 108.003},
        {"id": "P002", "name": "P2", "kind": "poi", "score": 7.0, "lat": 16.006, "lon": 108.003},
    ]
    params = json.loads(Path(PARAMS_FILE).read_text(encoding="utf-8"))
    graph = GraphData(nodes, edges, points, params, widths=road_widths(widths_path))
    orig_coords = np.array([[108.003, 16.0005]])
    return {
        "graph": graph,
        "nodes_path": nodes_path,
        "edges_path": edges_path,
        "widths_path": widths_path,
        "points": points,
        "orig_coords": orig_coords,
    }


def test_edge_dijkstra_and_via_nodes_snapshot(edge_env: dict) -> None:
    """Edge Dijkstra tìm đường nhanh nhất và bung đúng ``via_nodes`` khi xuất snapshot."""
    graph = edge_env["graph"]
    walk = edge_dijkstra(graph, "H01", "H02", budget_min=10.0, start_minute=480)
    assert walk.nodes == (0, 1)
    assert walk.score == 0.0
    assert walk.narrow_m == pytest.approx(600.0)

    snap = walk_to_snapshot(graph, walk, "H01", "H02", orig_coords=edge_env["orig_coords"])
    assert snap["route"] == ["H01", "H02"]
    assert snap["geometry"]["coordinates"] == [
        [108.0, 16.0],
        [108.003, 16.0005],
        [108.006, 16.0],
    ]
    assert len(snap["instructions"]) == 1
    assert snap["instructions"][0]["distance_m"] == pytest.approx(600.0)


@pytest.mark.parametrize("algorithm", ["alns", "vns", "aco"])
def test_edge_metaheuristics_collect_pois_deterministically(edge_env: dict, algorithm: str) -> None:
    """Edge ALNS/VNS/ACO thu được cả hai POI, giữ đúng ngân sách và tái lập theo seed."""
    graph = edge_env["graph"]
    fn = {"alns": edge_alns, "vns": edge_vns, "aco": edge_aco}[algorithm]
    w1 = fn(graph, "H01", "H02", budget_min=10.0, start_minute=480, iterations=15, seed=7)
    w2 = fn(graph, "H01", "H02", budget_min=10.0, start_minute=480, iterations=15, seed=7)
    assert w1 == w2
    assert w1.nodes[0] == 0 and w1.nodes[-1] == 1
    assert w1.duration_s <= 600.0 + 1e-6
    assert w1.score == pytest.approx(12.0)

    snap = find_edge_route(
        edge_env["nodes_path"],
        edge_env["edges_path"],
        edge_env["widths_path"],
        edge_env["points"],
        "H01",
        "H02",
        algorithm=algorithm,
        budget_min=10.0,
        departure="08:00",
        iterations=15,
        seed=7,
        graph=graph,
        orig_coords=edge_env["orig_coords"],
    )
    assert snap["collected"] == ["P001", "P002"]
    assert snap["score"] == pytest.approx(12.0)


def test_prune_zero_poi_cycles_removes_redundant_loop(edge_env: dict) -> None:
    """Vòng lặp 0 -> 1 -> 0 -> 2 -> 1 bị cắt bỏ thành 0 -> 2 -> 1 mà vẫn giữ đủ POI."""
    graph = edge_env["graph"]
    edge_lookup = {
        (int(graph.src[e]), int(graph.dst[e])): int(e) for e in range(len(graph.dst))
    }
    loop_edges = [
        edge_lookup[(0, 1)],
        edge_lookup[(1, 0)],
        edge_lookup[(0, 2)],
        edge_lookup[(2, 1)],
    ]
    walk = evaluate_edge_walk(graph, 0, 1, loop_edges, 480, 3600.0)
    assert walk is not None
    pruned = _prune_zero_poi_cycles(graph, walk, 480, 3600.0)
    assert pruned.nodes == (0, 2, 1)
    assert pruned.score == walk.score
    assert pruned.duration_s < walk.duration_s


def test_edge_solvers_reject_infeasible_budget(edge_env: dict) -> None:
    """Ngân sách quá nhỏ phải ném ``NoRoadRouteError``."""
    with pytest.raises(NoRoadRouteError):
        edge_dijkstra(edge_env["graph"], "H01", "H02", budget_min=0.05)
    with pytest.raises(NoRoadRouteError):
        edge_alns(edge_env["graph"], "H01", "H02", budget_min=0.05, iterations=3)


@pytest.mark.parametrize("algorithm", ["alns", "vns", "aco"])
def test_edge_metaheuristics_parallel_workers(edge_env: dict, algorithm: str) -> None:
    """Chế độ song song ``workers=2`` chạy tất định và đạt chất lượng hợp lệ."""
    graph = edge_env["graph"]
    fn = {"alns": edge_alns, "vns": edge_vns, "aco": edge_aco}[algorithm]
    w1 = fn(
        graph, "H01", "H02", budget_min=10.0, start_minute=480, iterations=8, seed=3, workers=2
    )
    w2 = fn(
        graph, "H01", "H02", budget_min=10.0, start_minute=480, iterations=8, seed=3, workers=2
    )
    assert w1 == w2
    assert w1.nodes[0] == 0 and w1.nodes[-1] == 1
    assert w1.duration_s <= 600.0 + 1e-6
    assert w1.score == pytest.approx(12.0)


def test_edge_aco_ppo_return_rejects_narrow_shortcut(tmp_path: Path) -> None:
    """ACO ở mức cạnh tối ưu tổng phần thưởng PPO nên tránh đường hẹp dù ngắn hơn."""
    nodes = pd.DataFrame(
        [
            (0, 16.000, 108.000),
            (1, 16.000, 108.010),
            (2, 16.003, 108.005),
            (3, 16.005, 108.008),
        ],
        columns=["node_idx", "lat", "lon"],
    )
    edges = pd.DataFrame(
        [
            (0, 2, "To P", 450.0, ""),
            (2, 1, "Narrow Shortcut", 250.0, ""),
            (2, 3, "Wide Detour A", 450.0, ""),
            (3, 1, "Wide Detour B", 400.0, ""),
            (0, 1, "Unknown Direct", 1200.0, ""),
        ],
        columns=["from_idx", "to_idx", "road_name", "length_m", "via_nodes"],
    )
    widths_path = tmp_path / "widths.csv"
    widths_path.write_text(
        "Tên đường,Chiều rộng mặt (m)\n"
        "To P,8\n"
        "Narrow Shortcut,3.5\n"
        "Wide Detour A,8\n"
        "Wide Detour B,8\n",
        encoding="utf-8-sig",
    )
    points = [
        {"id": "H01", "name": "Start", "kind": "hub", "score": 0.0, "lat": 16.000, "lon": 108.000},
        {"id": "H02", "name": "End", "kind": "hub", "score": 0.0, "lat": 16.000, "lon": 108.010},
        {"id": "P001", "name": "P1", "kind": "poi", "score": 5.0, "lat": 16.003, "lon": 108.005},
    ]
    params = json.loads(Path(PARAMS_FILE).read_text(encoding="utf-8"))
    graph = GraphData(nodes, edges, points, params, widths=road_widths(widths_path))
    edge_lookup = {(int(graph.src[e]), int(graph.dst[e])): int(e) for e in range(len(graph.dst))}

    walk_narrow = evaluate_edge_walk(
        graph, 0, 1, [edge_lookup[(0, 2)], edge_lookup[(2, 1)]], 480, 1200.0
    )
    walk_wide = evaluate_edge_walk(
        graph,
        0,
        1,
        [edge_lookup[(0, 2)], edge_lookup[(2, 3)], edge_lookup[(3, 1)]],
        480,
        1200.0,
    )
    walk_unknown = evaluate_edge_walk(graph, 0, 1, [edge_lookup[(0, 1)]], 480, 1200.0)
    assert walk_narrow is not None and walk_wide is not None and walk_unknown is not None

    # Theo thước đo quality cũ (weighted_m), đường tắt hẹp (950 m) thắng đường rộng (1300 m);
    # nhưng theo tổng phần thưởng BusRouteEnv của PPO, đường rộng thắng rõ rệt.
    assert walk_narrow.quality > walk_wide.quality
    assert _walk_ppo_return(graph, walk_wide, 480) > _walk_ppo_return(graph, walk_narrow, 480)
    assert walk_unknown.unknown_width_m == pytest.approx(1200.0)
    # Đường unknown_width 1200 m chịu phạt -0.05 * 1.2 = -0.06 ngoài time_penalty và success_bonus.
    expected_unknown_return = 1.0 - 0.05 * 1.2 - 0.01 * (walk_unknown.duration_s / 60.0)
    assert _walk_ppo_return(graph, walk_unknown, 480) == pytest.approx(expected_unknown_return)

    chosen = edge_aco(graph, "H01", "H02", budget_min=20.0, start_minute=480, iterations=12, seed=0)
    assert chosen.score == pytest.approx(5.0)
    assert chosen.narrow_m == pytest.approx(0.0)
    assert chosen.nodes == (0, 2, 3, 1)


def test_nearest_poi_at_matches_to_nearest_with_duplicate_and_empty_subsets(
    tmp_path: Path,
) -> None:
    """``GraphData.nearest_poi_at`` khớp tuyệt đối ``to_nearest`` trên mọi tập POI con."""
    nodes = pd.DataFrame(
        [
            (0, 16.000, 108.000),
            (1, 16.000, 108.010),
            (2, 16.003, 108.004),
            (3, 16.005, 108.007),
        ],
        columns=["node_idx", "lat", "lon"],
    )
    edges = pd.DataFrame(
        [
            (0, 2, "A", 400.0, ""),
            (2, 3, "B", 350.0, ""),
            (3, 1, "C", 500.0, ""),
            (0, 1, "D", 1400.0, ""),
        ],
        columns=["from_idx", "to_idx", "road_name", "length_m", "via_nodes"],
    )
    points = [
        {"id": "H01", "name": "Start", "kind": "hub", "score": 0.0, "lat": 16.000, "lon": 108.000},
        {"id": "H02", "name": "End", "kind": "hub", "score": 0.0, "lat": 16.000, "lon": 108.010},
        {"id": "P001", "name": "P1", "kind": "poi", "score": 5.0, "lat": 16.003, "lon": 108.004},
        {"id": "P002", "name": "P2 dup", "kind": "poi", "score": 3.0, "lat": 16.003, "lon": 108.004},
        {"id": "P003", "name": "P3", "kind": "poi", "score": 7.0, "lat": 16.005, "lon": 108.007},
    ]
    params = json.loads(Path(PARAMS_FILE).read_text(encoding="utf-8"))
    graph = GraphData(nodes, edges, points, params)
    all_nodes = np.arange(graph.n, dtype=np.int64)

    for mask in (0, 0b001, 0b011, 0b100, 0b111):
        active_rows = graph.active_poi_rows(mask, end_node=1)
        left = graph.poi_node[
            [
                i
                for i in range(len(graph.poi_node))
                if not (mask & (1 << i)) and int(graph.poi_node[i]) != 1
            ]
        ]
        expected = graph.to_nearest("morning", left)
        got = graph.nearest_poi_at("morning", all_nodes, active_rows)
        assert np.allclose(got, expected, equal_nan=True)
        for node_idx in range(graph.n):
            scalar_got = graph.nearest_poi_at("morning", node_idx, active_rows)
            if np.isinf(expected[node_idx]):
                assert np.isinf(scalar_got)
            else:
                assert scalar_got == pytest.approx(expected[node_idx])


def test_walk_ppo_return_matches_bus_route_env_step_by_step(edge_env: dict) -> None:
    """``_walk_ppo_return`` khớp tuyệt đối tổng phần thưởng từng bước của ``BusRouteEnv``."""
    graph: GraphData = edge_env["graph"]
    edge_lookup = {(int(graph.src[e]), int(graph.dst[e])): int(e) for e in range(len(graph.dst))}
    # Chọn lộ trình 0 -> 2 -> 3 -> 2 -> 1 (có revisit tại nút 2 và thu cả 2 POI).
    edges = [
        edge_lookup[(0, 2)],
        edge_lookup[(2, 3)],
        edge_lookup[(3, 2)],
        edge_lookup[(2, 1)],
    ]
    walk = evaluate_edge_walk(graph, 0, 1, edges, 480, 1200.0)
    assert walk is not None

    env = BusRouteEnv(graph)
    env.reset(seed=0, options={"start": "H01", "end": "H02", "start_minute": 480, "budget_s": 1200.0})
    total_env_reward = 0.0
    for edge_id in edges:
        lo, hi = int(graph.indptr[env.cur]), int(graph.indptr[env.cur + 1])
        action = int(edge_id - lo)
        assert 0 <= action < (hi - lo)
        _, r, terminated, truncated, info = env.step(action)
        total_env_reward += r
        if terminated or truncated:
            assert info["success"] == 1.0

    assert _walk_ppo_return(graph, walk, 480) == pytest.approx(total_env_reward)


def test_prune_zero_poi_cycles_caches_current_rank(edge_env: dict) -> None:
    """``_prune_zero_poi_cycles`` chỉ gọi ``rank_fn`` 1 lần cho trạng thái ban đầu và 1 lần/vòng."""
    graph = edge_env["graph"]
    edge_lookup = {(int(graph.src[e]), int(graph.dst[e])): int(e) for e in range(len(graph.dst))}
    loop_edges = [
        edge_lookup[(0, 1)],
        edge_lookup[(1, 0)],
        edge_lookup[(0, 2)],
        edge_lookup[(2, 1)],
    ]
    walk = evaluate_edge_walk(graph, 0, 1, loop_edges, 480, 3600.0)
    assert walk is not None

    calls = 0

    def counting_rank(w):
        nonlocal calls
        calls += 1
        return w.quality

    pruned = _prune_zero_poi_cycles(graph, walk, 480, 3600.0, rank_fn=counting_rank)
    assert pruned.nodes == (0, 2, 1)
    # 1 lần cho trạng thái ban đầu + 1 lần cho ứng viên sau khi cắt 1 vòng = 2 lần (không tính lại current).
    assert calls == 2


def test_slot_of_lookup_matches_window_definitions() -> None:
    """Bảng tra ``slot_of`` O(1) khớp hoàn toàn định nghĩa ``TIME_SLOTS`` quanh ranh giới ngày."""
    def reference_slot(minute: int) -> str:
        for name, (s, e) in TIME_SLOTS.items():
            if to_minutes(s) <= minute < to_minutes(e):
                return name
        return "night"

    for minute in range(-10, 1450):
        assert slot_of(minute) == reference_slot(minute)


def test_oop_solvers_shared_base_contracts(edge_env: dict, tmp_path: Path) -> None:
    """Các lớp kế thừa ``BaseRouteSolver`` chia sẻ cùng ``RoadDatasetSpec``, ``RouteRequest`` và ``RouteOutput``."""
    data = RoadDatasetSpec.from_raw(
        edge_env["nodes_path"],
        edge_env["edges_path"],
        edge_env["widths_path"],
        edge_env["points"],
        reduced_dir=tmp_path / "empty_reduced",
        use_reduced=False,
    )
    request = RouteRequest.from_departure(
        "H01", "H02", 10.0, departure="08:00", iterations=10, seed=7
    )

    for solver in (
        StreetDijkstraSolver(),
        StreetALNSSolver(),
        StreetACOSolver(),
        PoiALNSSolver(),
        PoiACOSolver(),
    ):
        output = solver.run(request, data=data)
        assert isinstance(output, RouteOutput)
        snap = output.to_snapshot()
        assert snap["route"][0] == "H01" and snap["route"][-1] == "H02"
        assert snap["duration_min"] <= 10.0 + 1e-6
        assert snap["instructions"]
        assert snap["geometry"]["type"] == "LineString"

