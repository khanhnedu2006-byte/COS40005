"""Test the Waypoint PPO action and checkpoint contracts."""

import json
from queue import Queue
from threading import Event
from zipfile import ZipFile

import pandas as pd
import pytest
import torch
from stable_baselines3.common.env_checker import check_env
from test_road_search import graph_fixture, point

from busroute.routing.ppo_waypoint import infer, load_policy, train
from busroute.routing.road_graph import NoRoadRouteError, RoadGraph
from busroute.routing.waypoint_env import (
    BENCHMARK_CASES,
    TRAIN_BUDGETS,
    WaypointEnv,
    WaypointFeatures,
    WaypointTrip,
)
from Web.app import RunRequest, create_app, search_worker


def test_action_mask_keeps_finish_and_preserves_the_verified_edges():
    features = WaypointFeatures(graph_fixture())
    assert features.action_ids == ("P", "Q", None)
    trip = WaypointTrip(features, ("A", "B", 480, 0.95))
    assert trip.observe()[-3:].tolist() == [1, 0, 1]
    with pytest.raises(ValueError, match="masked"):
        trip.step(1)
    assert trip.step(0) == 0.5
    assert trip.observe()[-3:].tolist() == [0, 0, 1]
    assert trip.step(features.finish) == 0
    result = trip.snapshot()
    assert result["route"] == ["A", "P", "B"]
    assert result["score"] == 5
    assert result["collected"] == ["P"]
    assert result["duration_min"] == pytest.approx(0.9)

    cached_rows = len(features.connections.rows)
    assert WaypointTrip(features, ("A", "B", 480, 0.95)).observe()[-3:].tolist() == [
        1,
        0,
        1,
    ]
    assert len(features.connections.rows) == cached_rows
    with pytest.raises(NoRoadRouteError, match="budget"):
        WaypointTrip(features, ("A", "B", 480, 0.5))
    with pytest.raises(NoRoadRouteError, match="No directed route"):
        WaypointTrip(features, ("B", "A", 480, 10))


@pytest.mark.parametrize("departure, finish_seconds", [(629, 150), (1439, 75)])
def test_return_is_reserved_at_arrival_clock(departure, finish_seconds):
    graph = graph_fixture()
    graph.times["morning"][:] = [50, 60, 15, 300, 40, 100]
    graph.times["ban_noon"][:] = [50, 60, 90, 300, 40, 100]
    graph.times["night"][:] = [50, 60, 15, 300, 40, 100]
    features = WaypointFeatures(graph)
    trip = WaypointTrip(features, ("A", "B", departure, finish_seconds / 60))
    assert trip.observe()[-3] == 1
    trip.step(0)
    assert trip.elapsed == 60
    assert trip.reserve.arrival == finish_seconds
    trip.observe()
    trip.step(features.finish)
    assert trip.snapshot()["duration_min"] == pytest.approx(finish_seconds / 60)


def test_terminal_sink_masks_outbound_paths_through_destination():
    fixture = graph_fixture()
    edges = pd.DataFrame(
        [(0, 1, "Direct", 600), (1, 2, "Past finish", 450), (2, 1, "Return", 450)],
        columns=["from_idx", "to_idx", "road_name", "length_m"],
    )
    graph = RoadGraph(
        fixture.nodes,
        edges,
        {},
        list(fixture.points.values()),
        time_params=fixture.time_params,
    )
    features = WaypointFeatures(graph)
    trip = WaypointTrip(features, ("A", "B", 480, 10))
    assert trip.observe()[-3:].tolist() == [0, 0, 1]
    trip.step(features.finish)
    assert trip.snapshot()["route"] == ["A", "B"]


def test_training_excludes_ordered_benchmark_pairs_at_every_time_and_budget():
    fixture = graph_fixture()
    points = [point("H01", 16, 108), point("H02", 16, 108.006), point("H03", 16, 108)]
    graph = RoadGraph(
        fixture.nodes, fixture.edges, {}, points, time_params=fixture.time_params
    )
    cases = WaypointFeatures(graph).training_cases()
    heldout = {case[:2] for case in BENCHMARK_CASES}
    assert all(case[:2] not in heldout for case in cases)
    assert {case[3] for case in cases} == set(TRAIN_BUDGETS)
    assert len({case[2] for case in cases}) == 5


def test_waypoint_checkpoint_training_and_web_worker(tmp_path):
    fixture = graph_fixture()
    files = [
        tmp_path / name for name in ("nodes.parquet", "edges.parquet", "widths.csv")
    ]
    fixture.nodes.to_parquet(files[0])
    fixture.edges.to_parquet(files[1])
    files[2].write_text("Tên đường,Chiều rộng mặt (m)\nDirect,4\n", encoding="utf-8")
    points = list(fixture.points.values())
    graph = RoadGraph.from_files(*files, points)
    features = WaypointFeatures(graph)
    checkpoint = tmp_path / "waypoint.zip"
    saved = train(
        features,
        checkpoint,
        steps=6,
        rollout_steps=4,
        batch_size=2,
        epochs=1,
        threads=1,
    )
    assert saved.num_timesteps == 8
    model = load_policy(checkpoint, features)
    scenario = ("A", "B", 480, 10)
    trip = WaypointTrip(features, scenario)
    obs = trip.observe()
    obs[-3] = 0.0
    with torch.no_grad():
        distribution = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0])
    assert distribution.distribution.probs[0, 0] == 0.0

    body = RunRequest(
        start="A",
        end="B",
        algorithm="ppo_waypoint",
        budget_min=10,
        checkpoint=str(checkpoint),
    )
    output = Queue()
    search_worker(files, points, body.model_dump(), output, Event())
    outcome = output.get_nowait()
    assert outcome["status"] == "result", outcome
    assert outcome["result"] == infer(features, model, scenario)
    assert outcome["result"]["instructions"]
    schema = create_app(points=points).openapi()
    assert (
        "ppo_waypoint"
        in schema["components"]["schemas"]["RunRequest"]["properties"]["algorithm"][
            "enum"
        ]
    )

    graph.times["morning"][0] += 1
    with pytest.raises(ValueError, match="does not match"):
        load_policy(checkpoint, features)
    edge_checkpoint = tmp_path / "edge.zip"
    with ZipFile(edge_checkpoint, "w") as archive:
        archive.writestr("data", json.dumps({"road_metadata": {"version": 3}}))
    with pytest.raises(ValueError, match="Waypoint PPO"):
        load_policy(edge_checkpoint, features)


def test_waypoint_env_passes_gym_check_and_validates_training_and_metadata(
    tmp_path,
) -> None:
    """Verify Gym check_env compliance, progress flush, and input validation."""
    features = WaypointFeatures(graph_fixture())
    env = WaypointEnv(features, (("A", "B", 480, 0.95),))
    check_env(env)
    with pytest.raises(RuntimeError, match="Reset"):
        WaypointEnv(features, (("A", "B", 480, 0.95),)).step(0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (features.obs_size,)
    _, reward, terminated, truncated, info = env.step(1)
    assert terminated and not truncated and reward == -1.0 and info["invalid_action"]
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(features.finish)

    fixture = graph_fixture()
    points = [point("H01", 16, 108), point("H02", 16, 108.006), point("H03", 16, 108)]
    graph = RoadGraph(
        fixture.nodes, fixture.edges, {}, points, time_params=fixture.time_params
    )
    train_features = WaypointFeatures(graph)
    records = []
    ckpt = tmp_path / "wp_progress.zip"
    train(
        train_features,
        ckpt,
        steps=4,
        rollout_steps=4,
        batch_size=2,
        epochs=1,
        threads=1,
        progress=records.append,
    )
    assert any(record.get("steps") == 4 for record in records)

    bad_ckpt = tmp_path / "bad_threads.zip"
    with ZipFile(ckpt) as src, ZipFile(bad_ckpt, "w") as dst:
        for name in src.namelist():
            content = src.read(name)
            if name == "data":
                payload = json.loads(content)
                payload["waypoint_metadata"]["threads"] = True
                content = json.dumps(payload).encode()
            dst.writestr(name, content)
    with pytest.raises(ValueError, match="threads"):
        load_policy(bad_ckpt, train_features)

    with pytest.raises(ValueError, match="Seed"):
        train(train_features, tmp_path / "neg.zip", steps=4, rollout_steps=4, batch_size=2, seed=-1)
    with pytest.raises(ValueError, match="Seed"):
        train(train_features, tmp_path / "bool.zip", steps=4, rollout_steps=4, batch_size=2, seed=True)
