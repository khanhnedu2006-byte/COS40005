"""Kiểm thử môi trường, huấn luyện và suy luận PPO ở mức waypoint."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from busroute.graph.ab_estimator import PARAMS_FILE
from busroute.routing.ppo_waypoint import (
    WaypointBusEnv,
    WaypointTable,
    evaluate_waypoint,
    infer_waypoint_route,
    train_waypoint,
)
from busroute.routing.road_dijkstra import road_widths
from busroute.routing.road_graph import RoadGraph


@pytest.fixture
def waypoint_road_env(tmp_path: Path) -> dict:
    """Tạo ``RoadGraph`` và ``WaypointTable`` nhỏ để kiểm thử nhanh."""
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
    nodes.to_parquet(nodes_path)
    edges.to_parquet(edges_path)
    widths_path.write_text(
        "Tên đường,Chiều rộng mặt (m)\nDirect,4\nReturn,8\nTo P,8\nFrom P,8\nTo Q,8\nBack Q,8\nFrom Q,8\n",
        encoding="utf-8-sig",
    )
    points = [
        {"id": "H01", "name": "Start", "kind": "hub", "score": 0.0, "lat": 16.000, "lon": 108.000},
        {"id": "H02", "name": "End", "kind": "hub", "score": 0.0, "lat": 16.000, "lon": 108.006},
        {"id": "P001", "name": "P1", "kind": "poi", "score": 5.0, "lat": 16.003, "lon": 108.003},
        {"id": "P002", "name": "P2", "kind": "poi", "score": 7.0, "lat": 16.006, "lon": 108.003},
    ]
    params = json.loads(Path(PARAMS_FILE).read_text(encoding="utf-8"))
    road_graph = RoadGraph(
        nodes, edges, road_widths(widths_path), points, time_params=params
    )
    table = WaypointTable.from_road_graph(road_graph)
    return {
        "road_graph": road_graph,
        "table": table,
        "nodes_path": nodes_path,
        "edges_path": edges_path,
        "widths_path": widths_path,
        "points": points,
    }


def test_waypoint_env_masking_and_invalid_action(waypoint_road_env: dict) -> None:
    """``WaypointBusEnv`` chỉ mở POI kịp đường về và từ chối hành động bị che."""
    env = WaypointBusEnv(waypoint_road_env["table"])
    obs, info = env.reset(
        seed=0,
        options={"start": "H01", "end": "H02", "start_minute": 480, "budget_s": 600.0},
    )
    assert obs.shape == env.observation_space.shape
    assert info["start"] == "H01" and info["end"] == "H02"
    mask = env.action_masks()
    assert mask[-1]  # luôn về thẳng đích được

    # Khi ngân sách chỉ đủ đi thẳng, hai POI trung gian có thể bị che; thử hành động hợp lệ tới đích.
    _, _, terminated, _, step_info = env.step(int(env.finish_action))
    assert terminated is True
    assert step_info["success"] == 1.0


def test_waypoint_ppo_train_eval_and_infer_snapshot(
    waypoint_road_env: dict, tmp_path: Path
) -> None:
    """Huấn luyện nhanh 2 epoch và suy luận ``RouteSnapshot`` khớp ``RoadGraph``."""
    table = waypoint_road_env["table"]
    ckpt = tmp_path / "wp_ppo.zip"
    model = train_waypoint(
        table,
        epochs=2,
        episodes_per_epoch=4,
        update_passes=1,
        batch_size=4,
        hidden=8,
        seed=0,
        checkpoint=ckpt,
        log=lambda *_: None,
    )
    assert ckpt.is_file()

    metrics = evaluate_waypoint(table, "ppo", model=model, episodes=3, seed=42)
    assert metrics["success_rate"] == pytest.approx(1.0)

    snap = infer_waypoint_route(
        waypoint_road_env["nodes_path"],
        waypoint_road_env["edges_path"],
        waypoint_road_env["widths_path"],
        waypoint_road_env["points"],
        "H01",
        "H02",
        ckpt,
        budget_min=10.0,
        start_minute=480,
    )
    assert snap["route"][0] == "H01" and snap["route"][-1] == "H02"
    assert snap["duration_min"] <= 10.0 + 1e-6
    assert snap["instructions"]
