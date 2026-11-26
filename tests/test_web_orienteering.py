"""Road-search API, isolated jobs, cancellation and final-only browser contracts."""

import queue
from pathlib import Path
from typing import Any
import re
import threading
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from Web.app import create_app, search_worker
from busroute.routing.road_dijkstra import find_route


@pytest.fixture
def road_data(tmp_path):
    nodes = pd.DataFrame(
        [(0, 16.0, 108.0), (1, 16.0, 108.006), (2, 16.003, 108.003)],
        columns=["node_idx", "lat", "lon"],
    )
    edges = pd.DataFrame(
        [(0, 1, "Direct", 600), (0, 2, "To P", 450), (2, 1, "From P", 450)],
        columns=["from_idx", "to_idx", "road_name", "length_m"],
    )
    files = tuple(
        tmp_path / filename
        for filename in ["nodes.parquet", "edges.parquet", "widths.csv"]
    )
    nodes.to_parquet(files[0])
    edges.to_parquet(files[1])
    files[2].write_text(
        "Tên đường,Chiều rộng mặt (m)\nDirect,4\nTo P,8\nFrom P,8\n",
        encoding="utf-8-sig",
    )
    points = [
        dict(id=id, name=id, kind=kind, score=score, lat=lat, lon=lon)
        for id, kind, score, lat, lon in [
            ("A", "hub", 0, 16.0, 108.0),
            ("B", "hub", 0, 16.0, 108.006),
            ("P", "poi", 5, 16.003, 108.003),
            ("Far", "poi", 9, 17.0, 109.0),
        ]
    ]
    return files, points


def make_app(data):
    files, points = data
    return create_app(
        nodes_path=files[0], edges_path=files[1], widths_path=files[2], points=points
    )


@pytest.fixture
def client(road_data):
    with TestClient(make_app(road_data)) as session:
        yield session


def request(**changes):
    return {
        **dict(
            start="A",
            end="B",
            algorithm="alns",
            budget_min=6,
            departure="08:00",
            iterations=5,
            seed=7,
        ),
        **changes,
    }


def wait_for_result(client, id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        response = client.get("/api/runs/" + id)
        assert response.status_code == 200
        result = response.json()
        if result["status"] != "running":
            return result
        assert result["result"] is None
        time.sleep(0.02)
    pytest.fail("solver did not finish")


@pytest.mark.parametrize("algorithm", ["dijkstra", "alns", "vns", "aco"])
def test_final_result_matches_core_solver_and_download(client, road_data, algorithm):
    params = request(algorithm=algorithm)
    response = client.post("/api/runs", json=params)
    assert response.status_code == 201
    id = response.json()["id"]
    outcome = wait_for_result(client, id)
    assert outcome["status"] == "result", outcome
    files, points = road_data
    expected = find_route(
        *files,
        points,
        "A",
        "B",
        algorithm=algorithm,
        budget_min=6,
        departure="08:00",
        iterations=5,
        seed=7,
    )
    assert outcome["result"] == expected
    assert client.get(f"/api/runs/{id}/result").json() == expected
    assert client.get(f"/api/runs/{id}").json() == outcome
    assert client.get(f"/api/runs/{id}/events").status_code == 404
    assert not hasattr(client.app.state.lab.run, "events")
    assert "instructions" in expected and "geometry" in expected
    assert expected["route"] == (
        ["A", "B"] if algorithm == "dijkstra" else ["A", "P", "B"]
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"algorithm": "unknown"},
        {"algorithm": "ppo"},
        {"start": "unknown"},
        {"end": "A"},
        {"start": "Far"},
        {"budget_min": 0},
        {"budget_min": -1},
        {"budget_min": "Infinity"},
        {"iterations": -1},
        {"iterations": 1.2},
        {"seed": "oops"},
        {"departure": "25:00"},
        {"departure": "08:99"},
        {"budget_km": 30},
    ],
)
def test_invalid_input(client, changes):
    assert client.post("/api/runs", json=request(**changes)).status_code == 422


def test_stop_conflict_and_repeat(client):
    id = client.post("/api/runs", json=request(iterations=1000000)).json()["id"]
    assert client.post("/api/runs", json=request()).status_code == 409
    assert client.get(f"/api/runs/{id}/result").status_code == 409
    assert client.post(f"/api/runs/{id}/stop").json()["status"] == "stopped"
    assert wait_for_result(client, id) == dict(
        status="stopped", result=None, message=None
    )
    next_id = client.post("/api/runs", json=request(iterations=0)).json()["id"]
    assert wait_for_result(client, next_id)["status"] == "result"
    assert client.get(f"/api/runs/{id}/result").status_code == 404


def test_infeasible_route_reports_error_then_allows_new_run(client):
    id = client.post("/api/runs", json=request(budget_min=0.1)).json()["id"]
    outcome = wait_for_result(client, id)
    assert outcome["status"] == "error" and "budget" in outcome["message"]
    assert outcome["result"] is None
    next_id = client.post("/api/runs", json=request()).json()["id"]
    assert wait_for_result(client, next_id)["status"] == "result"


def test_bootstrap_snapping_defaults_assets_and_solver_contract(client):
    bootstrap = client.get("/api/bootstrap").json()
    assert bootstrap["defaults"]["budget_min"] == 240
    assert bootstrap["defaults"]["start"] != bootstrap["defaults"]["end"]
    assert bootstrap["status"]["graph"] == "ready"
    far = next(p for p in bootstrap["points"] if p["id"] == "Far")
    assert not far["routing_available"] and far["snap_distance_m"] > 300
    assert client.get("/assets/vendor/maplibre-gl.js").status_code == 200
    schema = client.get("/openapi.json").json()["components"]["schemas"]
    assert set(schema["RunRequest"]["properties"]["algorithm"]["enum"]) == {
        "alns",
        "vns",
        "aco",
        "dijkstra",
        "ppo",
    }
    assert "distance_km" in schema["RouteSnapshot"]["required"]
    assert "instructions" in schema["RouteSnapshot"]["required"]


def test_missing_graph_has_readable_status_and_rejects_search(road_data):
    files, points = road_data
    files[1].unlink()
    with TestClient(make_app(road_data)) as client:
        status = client.get("/api/status").json()
        assert status["graph"] == "error" and "Missing road files" in status["message"]
        assert client.post("/api/runs", json=request()).status_code == 503


@pytest.mark.parametrize("algorithm", ["alns", "vns", "aco", "dijkstra"])
def test_worker_sends_one_terminal_message_and_obeys_cancellation(road_data, algorithm):
    files, points = road_data
    output = queue.Queue()
    cancelled = threading.Event()
    search_worker(files, points, request(algorithm=algorithm), output, cancelled)
    assert output.qsize() == 1 and output.get()["status"] == "result"
    cancelled.set()
    search_worker(files, points, request(algorithm=algorithm), output, cancelled)
    assert output.empty()


def test_final_only_browser_shell(client):
    html = client.get("/").text
    script = client.get("/assets/app.js").text
    for id in [
        "route-guide",
        "guide-stops",
        "road-directions-list",
        "budget_min",
        "departure",
    ]:
        assert f'id="{id}"' in html
    for obsolete in ["timeline", "inspector", "score-chart", "play", "budget_km"]:
        assert f'id="{obsolete}"' not in html
    assert "EventSource" not in script and "requestAnimationFrame" not in script
    assert "option.disabled = point.routing_available === false" in script
    assert '<option value="aco">' in html
    assert '<option value="ppo">' in html
    assert 'id="checkpoint"' in html
    assert 'value="checkpoints/ppo_reduced.zip"' in html


def test_trained_road_ppo_checkpoint_in_web(
    client: TestClient,
    road_data: tuple[tuple[Path, Path, Path], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    """Verify an SB3 reduced-graph PPO ZIP loads through the web worker and shared renderer."""
    pytest.importorskip("sb3_contrib")
    import json

    from busroute.graph.ab_estimator import PARAMS_FILE
    from busroute.routing.ppo_env import GraphData
    from busroute.routing.ppo_train import train
    from busroute.routing.road_dijkstra import road_widths

    files, points = road_data
    params = json.loads(Path(PARAMS_FILE).read_text(encoding="utf-8"))
    graph = GraphData(
        pd.read_parquet(files[0]),
        pd.read_parquet(files[1]),
        points,
        params,
        widths=road_widths(files[2]),
    )
    path = tmp_path / "ppo_reduced.zip"
    train(
        graph,
        epochs=1,
        episodes_per_epoch=2,
        update_passes=1,
        batch_size=4,
        hidden=8,
        max_episode_steps=12,
        checkpoint=path,
        log=lambda *_: None,
    )
    params_req = request(algorithm="ppo", checkpoint=str(path))
    response = client.post("/api/runs", json=params_req)
    assert response.status_code == 201
    outcome = wait_for_result(client, response.json()["id"])
    assert outcome["status"] == "result", outcome
    assert outcome["result"] == find_route(
        *files,
        points,
        "A",
        "B",
        algorithm="ppo",
        checkpoint=path,
        budget_min=6,
        departure="08:00",
    )
    assert outcome["result"]["instructions"]


def test_shell_refreshes_asset_urls_when_contents_change(
    road_data, tmp_path, monkeypatch
):
    import Web.app as web_app

    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "index.html").write_text(
        '<link href="/assets/app.css"><script src="/assets/app.js"></script>',
        encoding="utf-8",
    )
    (assets / "app.js").write_text("// old controls", encoding="utf-8")
    (assets / "app.css").write_text("body {}", encoding="utf-8")
    monkeypatch.setattr(web_app, "ASSETS", assets)
    with TestClient(make_app(road_data)) as client:
        first = client.get("/")
        assert first.headers["cache-control"] == "no-store"
        urls = re.findall(r'"(/assets/[^\"]+)"', first.text)
        assert len(urls) == 2 and all("?v=" in url for url in urls)
        assert all(client.get(url).status_code == 200 for url in urls)
        (assets / "app.js").write_text("// new dropdowns", encoding="utf-8")
        updated = re.findall(r'"(/assets/[^\"]+)"', client.get("/").text)
        assert updated[0] == urls[0] and updated[1] != urls[1]
        assert client.get(updated[1]).text == "// new dropdowns"
