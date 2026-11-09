"""Exercise real solver processes, event replay, and geometry API boundaries."""

import csv
import json
import threading
import time

import networkx as nx
import pytest
from shapely.geometry import LineString

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from busroute.routing.metaheuristics import ALNS, VNS, AntColony
from busroute.routing.od_evaluation import shortest_route
from busroute.routing.orienteering import ODMatrix
from Web.app import create_app
from Web.road_paths import RoadPaths


@pytest.fixture
def matrix_path(tmp_path):
    path = tmp_path / "matrix.csv"
    points = {
        "A": (16, 108),
        "B": (16, 108.003),
        "P": (16, 108.001),
        "Q": (16, 108.002),
    }
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "from_id",
                "to_id",
                "from_type",
                "to_type",
                "from_lat",
                "from_lon",
                "to_lat",
                "to_lon",
                "to_poi_score",
                "slot",
                "duration_s",
                "feasible",
                "pois_passed",
            ],
        )
        writer.writeheader()
        for a, (lat, lon) in points.items():
            for b, (blat, blon) in points.items():
                if a != b:
                    writer.writerow(
                        {
                            "from_id": a,
                            "to_id": b,
                            "from_type": "poi" if a in "PQ" else "hub",
                            "to_type": "poi" if b in "PQ" else "hub",
                            "from_lat": lat,
                            "from_lon": lon,
                            "to_lat": blat,
                            "to_lon": blon,
                            "to_poi_score": {"P": 3, "Q": 4}.get(b, 0),
                            "slot": "morning",
                            "duration_s": 60,
                            "feasible": 1,
                            "pois_passed": "",
                        }
                    )
    return path


def graph_loader(points):
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    for i, p in enumerate(points):
        graph.add_node(i, x=p["lon"], y=p["lat"])
    ids = {p["id"]: i for i, p in enumerate(points)}
    graph.add_edge(
        ids["A"],
        ids["P"],
        length=120,
        geometry=LineString([(108, 16), (108.0005, 16.0002), (108.001, 16)]),
    )
    return RoadPaths(graph, {p["id"]: (p["lat"], p["lon"]) for p in points})


@pytest.fixture
def client(matrix_path):
    with TestClient(create_app(matrix_path, graph_loader=graph_loader)) as session:
        yield session


def read_events(client, run_id, **kwargs):
    response = client.get(f"/api/runs/{run_id}/events", **kwargs)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def request(**changes):
    return {
        "start": "A",
        "end": "B",
        "algorithm": "alns",
        "departure": "08:00",
        "budget_min": 3,
        "iterations": 3,
        "seed": 7,
        **changes,
    }


def wait_for_graph(client):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status = client.get("/api/status").json()
        if status["graph"] != "loading":
            return status
        time.sleep(0.02)
    pytest.fail("graph did not initialize")


@pytest.mark.parametrize(
    "algorithm,iterations",
    [
        ("alns", 3),
        ("vns", 3),
        ("aco", 3),
        ("dijkstra", 3),
        ("alns", 0),
        ("aco", 0),
    ],
)
def test_process_results_and_reconnect_cursor(
    client, matrix_path, algorithm, iterations
):
    params = request(algorithm=algorithm, iterations=iterations)
    response = client.post("/api/runs", json=params)
    assert response.status_code == 201
    run_id = response.json()["id"]
    events = read_events(client, run_id)
    assert events[0]["event"] == "search_start"
    assert events[-1]["event"] == "result"
    assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
    assert any(e["event"] == "operator" for e in events)
    matrix = ODMatrix.from_csv(matrix_path)
    if algorithm == "dijkstra":
        expected = shortest_route(matrix, "A", "B", 480, 180)
        assert any(e.get("action") == "expand" for e in events)
    else:
        solver = {"alns": ALNS, "vns": VNS, "aco": AntColony}[algorithm]
        expected = solver(iterations=iterations, seed=7)(matrix, "A", "B", 480, 180)
        if algorithm == "aco" and iterations:
            assert {"ant_start", "insert", "pheromone_update"} <= {
                e.get("action") for e in events
            }
    assert events[-1]["best"] == expected.to_dict()
    assert client.get(f"/api/runs/{run_id}/result").json() == expected.to_dict()
    assert (
        read_events(client, run_id, headers={"Last-Event-ID": str(len(events) - 1)})
        == events[-1:]
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"algorithm": "ppo"},
        {"start": "unknown"},
        {"end": "A"},
        {"departure": "25:00"},
        {"departure": "09:60"},
        {"budget_min": 0},
        {"iterations": -1},
        {"iterations": 1.2},
        {"seed": "oops"},
    ],
)
def test_invalid_input(client, changes):
    assert client.post("/api/runs", json=request(**changes)).status_code == 422


def test_one_process_at_a_time_stop_and_repeat(client):
    run_id = client.post("/api/runs", json=request(iterations=100000)).json()["id"]
    assert client.post("/api/runs", json=request()).status_code == 409
    assert client.get(f"/api/runs/{run_id}/result").status_code == 409
    response = client.post(f"/api/runs/{run_id}/stop")
    assert response.status_code == 200 and response.json()["status"] == "stopped"
    assert read_events(client, run_id)[-1]["event"] == "stopped"
    next_id = client.post("/api/runs", json=request(iterations=0)).json()["id"]
    assert read_events(client, next_id)[-1]["event"] == "result"
    assert client.get(f"/api/runs/{run_id}/result").status_code == 404


def test_geometry_preserves_curves_and_caches_missing_directions(client):
    assert wait_for_graph(client)["graph"] == "ready"
    body = {"legs": [["A", "P"], ["P", "A"], ["A", "P"]]}
    response = client.post("/api/geometry", json=body)
    legs = response.json()["legs"]
    assert len(legs) == 2
    assert legs[0]["geometry"]["coordinates"][0][1] == [108.0005, 16.0002]
    assert legs[1]["geometry"] is None and legs[1]["message"]
    assert client.post("/api/geometry", json=body).json() == response.json()
    status = client.get("/api/status").json()
    assert status["cached_legs"] == 2 and status["graph_loads"] == 1
    assert (
        client.post("/api/geometry", json={"legs": [["unknown", "A"]]}).status_code
        == 422
    )


def test_graph_failure_does_not_hide_inputs_or_break_solver(matrix_path):
    def failed(_):
        raise RuntimeError("test graph unavailable")

    with TestClient(create_app(matrix_path, graph_loader=failed)) as client:
        assert wait_for_graph(client)["graph"] == "error"
        assert len(client.get("/api/bootstrap").json()["points"]) == 4
        assert (
            client.post("/api/geometry", json={"legs": [["A", "B"]]}).status_code == 503
        )
        run_id = client.post("/api/runs", json=request(iterations=0)).json()["id"]
        assert read_events(client, run_id)[-1]["event"] == "result"


def test_loading_graph_does_not_block_inputs_or_computation(matrix_path):
    release = threading.Event()

    def slow_loader(_):
        release.wait(timeout=10)

    try:
        with TestClient(create_app(matrix_path, graph_loader=slow_loader)) as client:
            bootstrap = client.get("/api/bootstrap").json()
            assert bootstrap["status"]["graph"] == "loading"
            run_id = client.post("/api/runs", json=request(iterations=0)).json()["id"]
            assert read_events(client, run_id)[-1]["event"] == "result"
            assert (
                client.post("/api/geometry", json={"legs": [["A", "B"]]}).status_code
                == 503
            )
            release.set()
            assert wait_for_graph(client)["graph_loads"] == 1
    finally:
        release.set()


def test_infeasible_search_emits_error_and_can_be_replaced(client):
    run_id = client.post("/api/runs", json=request(budget_min=0.1)).json()["id"]
    events = read_events(client, run_id)
    assert events[-1]["event"] == "error" and "initial route" in events[-1]["message"]
    assert client.get(f"/api/runs/{run_id}/result").status_code == 409
    next_id = client.post("/api/runs", json=request(iterations=0)).json()["id"]
    assert read_events(client, next_id)[-1]["event"] == "result"


def test_bootstrap_defaults_and_local_assets(client):
    bootstrap = client.get("/api/bootstrap").json()
    assert bootstrap["defaults"]["start"] != bootstrap["defaults"]["end"]
    assert bootstrap["defaults"]["start"] in {p["id"] for p in bootstrap["points"]}
    assert client.get("/").status_code == 200
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/vendor/maplibre-gl.js").status_code == 200
    assert client.get("/assets/vendor/maplibre-LICENSE.txt").status_code == 200
    assert client.get("/api/runs/missing/events").status_code == 404


def test_custom_matrix_with_only_h02_uses_distinct_defaults(matrix_path):
    with matrix_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields, rows = reader.fieldnames, list(reader)
    rename = {"A": "H02", "B": "Z", "P": "ZP", "Q": "ZQ"}
    for row in rows:
        row["from_id"], row["to_id"] = rename[row["from_id"]], rename[row["to_id"]]
    with matrix_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with TestClient(create_app(matrix_path, graph_loader=lambda _: None)) as client:
        defaults = client.get("/api/bootstrap").json()["defaults"]
        assert defaults["start"] == "H02" and defaults["end"] != "H02"


def test_dijkstra_expansion_snapshots_match_matrix_prefixes(client, matrix_path):
    matrix = ODMatrix.from_csv(matrix_path)
    run_id = client.post("/api/runs", json=request(algorithm="dijkstra")).json()["id"]
    events = read_events(client, run_id)
    expanded = [e for e in events if e.get("action") == "expand"]
    assert expanded[0]["candidate"]["route"] == ["A"]
    assert expanded[0]["candidate"]["duration_min"] == 0
    assert [e["expanded"] for e in expanded] == list(range(1, len(expanded) + 1))
    for event in expanded[1:]:
        prefix = event["candidate"]
        assert matrix.evaluate(prefix["route"], 480, 180).to_dict() == prefix


def test_openapi_exposes_all_solvers_and_typed_response_contracts(client):
    schema = client.get("/openapi.json").json()
    request_schema = schema["components"]["schemas"]["RunRequest"]
    assert set(request_schema["properties"]["algorithm"]["enum"]) == {
        "alns",
        "vns",
        "aco",
        "dijkstra",
    }
    assert "BootstrapPayload" in schema["components"]["schemas"]
    assert "GeometryPayload" in schema["components"]["schemas"]
