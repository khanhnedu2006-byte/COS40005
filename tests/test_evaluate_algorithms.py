"""Kiểm thử module và CLI đánh giá so sánh các thuật toán định tuyến."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_algorithms as eval_cli  # noqa: E402
from busroute.graph.ab_estimator import PARAMS_FILE  # noqa: E402
from busroute.routing import benchmark as benchmark_mod  # noqa: E402
from busroute.routing.benchmark import compare_algorithms, parse_modes  # noqa: E402
from busroute.routing.edge_metaheuristics import find_edge_route  # noqa: E402
from busroute.routing.ppo_env import GraphData  # noqa: E402
from busroute.routing.ppo_train import train  # noqa: E402
from busroute.routing.ppo_waypoint import WaypointTable, train_waypoint  # noqa: E402
from busroute.routing.road_dijkstra import find_route, road_widths  # noqa: E402
from busroute.routing.road_graph import RoadGraph  # noqa: E402


@pytest.fixture
def synthetic_road_env(tmp_path: Path) -> dict[str, Any]:
    """Tạo đồ thị đường nhỏ (Parquet + CSV bề rộng) và checkpoint PPO thử nghiệm."""
    pytest.importorskip("sb3_contrib")

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
            (0, 1, "Direct", 600.0),
            (1, 0, "Return", 600.0),
            (0, 2, "To P", 450.0),
            (2, 1, "From P", 450.0),
            (2, 3, "To Q", 330.0),
            (3, 2, "Back Q", 330.0),
            (3, 1, "From Q", 670.0),
            (0, 3, "To Q Direct", 900.0),
        ],
        columns=["from_idx", "to_idx", "road_name", "length_m"],
    )
    nodes_path = tmp_path / "nodes.parquet"
    edges_path = tmp_path / "edges.parquet"
    widths_path = tmp_path / "widths.csv"
    reduced_dir = tmp_path / "reduced"
    reduced_dir.mkdir()

    nodes.to_parquet(nodes_path)
    edges.to_parquet(edges_path)
    widths_path.write_text(
        "Tên đường,Chiều rộng mặt (m)\n"
        "Direct,4\n"
        "To P,8\n"
        "From P,8\n"
        "To Q,8\n"
        "From Q,8\n"
        "To Q Direct,8\n",
        encoding="utf-8-sig",
    )
    points = [
        {"id": "H01", "name": "Start Hub", "kind": "hub", "score": 0.0, "lat": 16.000, "lon": 108.000},
        {"id": "H02", "name": "End Hub", "kind": "hub", "score": 0.0, "lat": 16.000, "lon": 108.006},
        {"id": "P001", "name": "POI One", "kind": "poi", "score": 5.0, "lat": 16.003, "lon": 108.003},
        {"id": "P002", "name": "POI Two", "kind": "poi", "score": 7.0, "lat": 16.006, "lon": 108.003},
    ]

    params = json.loads(Path(PARAMS_FILE).read_text(encoding="utf-8"))
    graph = GraphData(nodes, edges, points, params, widths=road_widths(widths_path))
    ckpt_path = tmp_path / "ppo_test.zip"
    train(
        graph,
        epochs=1,
        episodes_per_epoch=3,
        update_passes=1,
        batch_size=4,
        hidden=8,
        seed=0,
        max_episode_steps=12,
        checkpoint=ckpt_path,
        log=lambda *_: None,
    )
    road_graph = RoadGraph(nodes, edges, road_widths(widths_path), points, time_params=params)
    wp_ckpt_path = tmp_path / "ppo_wp_test.zip"
    train_waypoint(
        WaypointTable.from_road_graph(road_graph),
        epochs=1,
        episodes_per_epoch=4,
        update_passes=1,
        batch_size=4,
        hidden=8,
        seed=0,
        checkpoint=wp_ckpt_path,
        log=lambda *_: None,
    )
    return {
        "nodes_path": nodes_path,
        "edges_path": edges_path,
        "widths_path": widths_path,
        "reduced_dir": reduced_dir,
        "points": points,
        "checkpoint": ckpt_path,
        "waypoint_checkpoint": wp_ckpt_path,
    }


def test_compare_algorithms_all_five_with_ppo_checkpoint(
    synthetic_road_env: dict[str, Any],
) -> None:
    """So sánh đủ 5 thuật toán ở chế độ street-reduced và đối chiếu khớp với ``find_edge_route`` / ``find_route``."""
    report = compare_algorithms(
        "H01",
        "H02",
        budget_min=10.0,
        departure="08:00",
        algorithms=("dijkstra", "alns", "vns", "aco", "ppo"),
        modes=("street-reduced",),
        checkpoint=synthetic_road_env["checkpoint"],
        iterations=10,
        seed=7,
        nodes_path=synthetic_road_env["nodes_path"],
        edges_path=synthetic_road_env["edges_path"],
        widths_path=synthetic_road_env["widths_path"],
        reduced_dir=synthetic_road_env["reduced_dir"],
        points=synthetic_road_env["points"],
    )

    assert report["config"]["start"] == "H01"
    assert report["config"]["end"] == "H02"
    assert report["config"]["departure_slot"] == "morning"
    assert report["config"]["modes"] == ["street-reduced"]
    assert [row["algorithm"] for row in report["results"]] == [
        "dijkstra",
        "alns",
        "vns",
        "aco",
        "ppo",
    ]

    for row in report["results"]:
        assert row["status"] == "ok", row
        assert row["mode"] == "street-reduced"
        expected = (
            find_route(
                synthetic_road_env["nodes_path"],
                synthetic_road_env["edges_path"],
                synthetic_road_env["widths_path"],
                synthetic_road_env["points"],
                "H01",
                "H02",
                algorithm="ppo",
                budget_min=10.0,
                departure="08:00",
                checkpoint=synthetic_road_env["checkpoint"],
            )
            if row["algorithm"] == "ppo"
            else find_edge_route(
                synthetic_road_env["nodes_path"],
                synthetic_road_env["edges_path"],
                synthetic_road_env["widths_path"],
                synthetic_road_env["points"],
                "H01",
                "H02",
                algorithm=row["algorithm"],
                budget_min=10.0,
                departure="08:00",
                iterations=10,
                seed=7,
                reduced_dir=synthetic_road_env["reduced_dir"],
            )
        )
        assert row["duration_min"] == pytest.approx(expected["duration_min"])
        assert row["distance_km"] == pytest.approx(expected["distance_km"])
        assert row["poi_score"] == pytest.approx(expected["score"])
        assert row["poi_count"] == len(expected["collected"])
        assert row["collected"] == expected["collected"]
        assert row["route"] == expected["route"]
        assert row["within_budget"] is True
        assert row["budget_util_pct"] == pytest.approx(expected["duration_min"] / 10.0 * 100.0)
        assert row["score_per_min"] == pytest.approx(expected["score"] / expected["duration_min"])
        assert row["score_per_km"] == pytest.approx(expected["score"] / expected["distance_km"])
        assert row["narrow_m"] == pytest.approx(expected["narrow_m"])
        assert row["narrow_pct"] == pytest.approx(
            expected["narrow_m"] / (expected["distance_km"] * 1000.0) * 100.0
        )
        assert row["avg_speed_kmh"] == pytest.approx(
            expected["distance_km"] / (expected["duration_min"] / 60.0)
        )
        assert row["solve_time_s"] >= 0.0
        assert row["detour_time_ratio"] is not None
        assert row["detour_dist_ratio"] is not None
        assert row["score_gain_vs_dijkstra"] is not None

    assert report["summary"]["best_poi_score"]["algorithm"] in {"alns", "vns", "aco"}
    assert report["summary"]["fastest_route"]["algorithm"] == "dijkstra"


def test_compare_algorithms_handles_missing_or_incompatible_checkpoint(
    synthetic_road_env: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Thiếu hoặc sai checkpoint chỉ đánh dấu lỗi riêng cho PPO, các thuật toán khác vẫn chạy."""
    missing = tmp_path / "does_not_exist.zip"
    report = compare_algorithms(
        "H01",
        "H02",
        budget_min=10.0,
        algorithms=("dijkstra", "alns", "ppo"),
        modes=("edge",),
        checkpoint=missing,
        iterations=5,
        nodes_path=synthetic_road_env["nodes_path"],
        edges_path=synthetic_road_env["edges_path"],
        widths_path=synthetic_road_env["widths_path"],
        reduced_dir=synthetic_road_env["reduced_dir"],
        points=synthetic_road_env["points"],
    )
    by_alg = {row["algorithm"]: row for row in report["results"]}
    assert by_alg["dijkstra"]["status"] == "ok"
    assert by_alg["alns"]["status"] == "ok"
    assert by_alg["ppo"]["status"] == "error"
    assert "not found" in by_alg["ppo"]["error"].lower()

    bad_ckpt = tmp_path / "corrupt.zip"
    bad_ckpt.write_text("not a zip", encoding="utf-8")
    report_bad = compare_algorithms(
        "H01",
        "H02",
        budget_min=10.0,
        algorithms=("dijkstra", "ppo"),
        modes=("edge",),
        checkpoint=bad_ckpt,
        nodes_path=synthetic_road_env["nodes_path"],
        edges_path=synthetic_road_env["edges_path"],
        widths_path=synthetic_road_env["widths_path"],
        reduced_dir=synthetic_road_env["reduced_dir"],
        points=synthetic_road_env["points"],
    )
    by_alg_bad = {row["algorithm"]: row for row in report_bad["results"]}
    assert by_alg_bad["dijkstra"]["status"] == "ok"
    assert by_alg_bad["ppo"]["status"] == "error"


def test_compare_algorithms_handles_infeasible_budget(
    synthetic_road_env: dict[str, Any],
) -> None:
    """Ngân sách quá nhỏ khiến không đi thẳng được sẽ báo lỗi từng dòng mà không crash."""
    report = compare_algorithms(
        "H01",
        "H02",
        budget_min=0.05,
        algorithms=("dijkstra", "alns", "ppo"),
        modes=("waypoint", "edge"),
        checkpoint=synthetic_road_env["checkpoint"],
        waypoint_checkpoint=synthetic_road_env["waypoint_checkpoint"],
        iterations=5,
        nodes_path=synthetic_road_env["nodes_path"],
        edges_path=synthetic_road_env["edges_path"],
        widths_path=synthetic_road_env["widths_path"],
        reduced_dir=synthetic_road_env["reduced_dir"],
        points=synthetic_road_env["points"],
    )
    assert all(row["status"] == "error" for row in report["results"])
    assert report["summary"]["best_poi_score"] is None


def test_cli_json_csv_export_and_custom_checkpoint(
    synthetic_road_env: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI nhận --checkpoint, xuất bảng so sánh ra stdout và ghi đúng JSON/CSV."""
    monkeypatch.setattr(
        eval_cli, "configured_points", lambda: synthetic_road_env["points"]
    )
    monkeypatch.setattr(
        benchmark_mod, "configured_points", lambda: synthetic_road_env["points"]
    )
    json_out = tmp_path / "out" / "report.json"
    csv_out = tmp_path / "out" / "report.csv"

    eval_cli.main(
        [
            "--mode",
            "edge",
            "--start",
            "H01",
            "--end",
            "H02",
            "--budget-min",
            "10",
            "--departure",
            "08:00",
            "--algorithms",
            "dijkstra,alns,ppo",
            "--checkpoint",
            str(synthetic_road_env["checkpoint"]),
            "--iterations",
            "5",
            "--nodes",
            str(synthetic_road_env["nodes_path"]),
            "--edges",
            str(synthetic_road_env["edges_path"]),
            "--widths",
            str(synthetic_road_env["widths_path"]),
            "--reduced-dir",
            str(synthetic_road_env["reduced_dir"]),
            "--json",
            str(json_out),
            "--csv",
            str(csv_out),
            "--show-pois",
        ]
    )

    captured = capsys.readouterr().out
    assert "Route Evaluation: H01 (Start Hub) -> H02 (End Hub)" in captured
    assert "dijkstra" in captured and "alns" in captured and "ppo" in captured
    assert "Highlights:" in captured
    assert "Routes & Collected POIs:" in captured
    assert "within 100m" not in captured
    assert "snapped to visited nodes" in captured

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert set(payload) == {"config", "timing", "results", "summary"}
    assert len(payload["results"]) == 3

    with csv_out.open(encoding="utf-8-sig", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert [r["algorithm"] for r in csv_rows] == ["dijkstra", "alns", "ppo"]
    assert "duration_min" in csv_rows[0]
    assert "poi_count" in csv_rows[0]
    assert "poi_score" in csv_rows[0]
    assert "distance_km" in csv_rows[0]


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--start", "H01", "--end", "H01", "--budget-min", "60"],
        ["--start", "H01", "--end", "H02", "--budget-min", "0"],
        ["--start", "H01", "--end", "H02", "--budget-min", "60", "--departure", "25:00"],
        ["--start", "H01", "--end", "H02", "--budget-min", "60", "--algorithms", "invalid_alg"],
        ["--start", "H01", "--end", "H02", "--budget-min", "60", "--mode", "invalid_mode"],
        ["--start", "H01", "--end", "H02", "--budget-min", "60", "--workers", "0"],
        ["--start", "H01", "--end", "H02", "--budget-min", "60", "--reduced-checkpoint", "nonexistent_ckpt.zip"],
        ["--start", "H01", "--end", "H02", "--budget-min", "60", "--poi-checkpoint", "nonexistent_wp.zip"],
        ["--start", "H01", "--end", "H02", "--budget-min", "60", "--full-checkpoint", "nonexistent_full.zip"],
    ],
)
def test_cli_rejects_invalid_args(
    bad_args: list[str],
    synthetic_road_env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tham số không hợp lệ hoặc --checkpoint chỉ định rõ nhưng không tồn tại phải thoát mã 2."""
    monkeypatch.setattr(
        eval_cli, "configured_points", lambda: synthetic_road_env["points"]
    )
    monkeypatch.setattr(
        benchmark_mod, "configured_points", lambda: synthetic_road_env["points"]
    )
    with pytest.raises(SystemExit) as exc:
        eval_cli.main(bad_args)
    assert exc.value.code == 2


def test_mode_aliases_and_3x5_matrix(
    synthetic_road_env: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ma trận 3×5 (poi, street-reduced, street-full × 5 thuật toán) và alias tên chế độ hoạt động đúng."""
    assert parse_modes("both") == ("poi", "street-reduced")
    assert parse_modes("waypoint,edge") == ("poi", "street-reduced")
    assert parse_modes("street") == ("street-reduced", "street-full")
    assert parse_modes("all") == ("poi", "street-reduced", "street-full")

    monkeypatch.setattr(
        eval_cli, "configured_points", lambda: synthetic_road_env["points"]
    )
    monkeypatch.setattr(
        benchmark_mod, "configured_points", lambda: synthetic_road_env["points"]
    )
    json_out = tmp_path / "matrix3x5.json"
    csv_out = tmp_path / "matrix3x5.csv"
    eval_cli.main(
        [
            "--mode",
            "all",
            "--start",
            "H01",
            "--end",
            "H02",
            "--budget-min",
            "10",
            "--departure",
            "08:00",
            "--poi-checkpoint",
            str(synthetic_road_env["waypoint_checkpoint"]),
            "--reduced-checkpoint",
            str(synthetic_road_env["checkpoint"]),
            "--full-checkpoint",
            str(synthetic_road_env["checkpoint"]),
            "--iterations",
            "5",
            "--workers",
            "2",
            "--nodes",
            str(synthetic_road_env["nodes_path"]),
            "--edges",
            str(synthetic_road_env["edges_path"]),
            "--widths",
            str(synthetic_road_env["widths_path"]),
            "--reduced-dir",
            str(synthetic_road_env["reduced_dir"]),
            "--json",
            str(json_out),
            "--csv",
            str(csv_out),
        ]
    )
    out = capsys.readouterr().out
    assert "poi" in out and "street-reduced" in out and "street-full" in out
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["config"]["workers"] == 2
    assert len(payload["results"]) == 15
    assert all(row["status"] == "ok" for row in payload["results"])
    assert [
        (row["mode"], row["algorithm"]) for row in payload["results"]
    ] == [
        (mode, alg)
        for mode in ("poi", "street-reduced", "street-full")
        for alg in ("dijkstra", "alns", "vns", "aco", "ppo")
    ]
    with csv_out.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 15
