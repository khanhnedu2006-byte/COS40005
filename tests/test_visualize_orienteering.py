"""Route tracing must describe the search without changing its results."""
import csv
import importlib
from pathlib import Path

import networkx as nx
import pytest
from shapely.geometry import LineString

from busroute.routing.metaheuristics import ALNS, VNS
from busroute.routing.orienteering import Leg, ODMatrix


@pytest.fixture
def viewer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("visualize_orienteering")


@pytest.mark.parametrize("solver", [ALNS, VNS])
def test_operator_trace_preserves_results_and_feasible_snapshots(solver):
    points = {"A", "B", "P", "Q"}
    matrix = ODMatrix({(a, b, "morning"): Leg(60, frozenset())
                       for a in points for b in points if a != b},
                      {"P": 3, "Q": 4}, points)
    events = []
    expected = solver(iterations=8, seed=9)(matrix, "A", "B", 480, 180)
    traced = solver(iterations=8, seed=9, trace=True, progress=events.append)
    assert traced(matrix, "A", "B", 480, 180) == expected
    assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
    assert {"insert", "remove"} <= {e.get("action") for e in events}
    assert any(e.get("accepted") is False for e in events)
    best_quality = []
    for event in events:
        for key in ("best", "current", "candidate"):
            if key in event:
                snapshot = event[key]
                assert matrix.evaluate(snapshot["route"], 480, 180).to_dict() == snapshot
        if "best" in event:
            best = event["best"]
            best_quality.append((best["score"], -best["duration_min"]))
    assert best_quality == sorted(best_quality)
    assert events[-1]["best"] == expected.to_dict()
    # Reusing an instance starts a new sequence, just like its random generator.
    events.clear()
    assert traced(matrix, "A", "B", 480, 180) == expected
    assert events[0]["sequence"] == 1


def test_road_paths_preserve_curves_direction_and_parallel_edge_choice(viewer):
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(1, x=108.0, y=16.0)
    graph.add_node(2, x=108.001, y=16.0)
    curve = [(108.0, 16.0), (108.0005, 16.0002), (108.001, 16.0)]
    graph.add_edge(1, 2, length=120, geometry=LineString(curve))
    graph.add_edge(1, 2, length=200, geometry=LineString([curve[0], curve[-1]]))
    paths = viewer.RoadPaths(graph, {"A": (16, 108), "B": (16, 108.001)})
    segments, missing = paths.route({"route": ["A", "B"]})
    assert segments == [curve] and missing == []
    assert paths.route({"route": ["B", "A"]}) == ([], ["B → A"])
    assert paths.leg("A", "B") is paths.leg("A", "B")


def test_distant_stop_does_not_draw_a_false_connection(viewer):
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(1, x=108.0, y=16.0)
    paths = viewer.RoadPaths(graph, {"A": (16, 108), "B": (17, 109)})
    assert paths.route({"route": ["A", "B"]}) == ([], ["A → B"])


def test_coordinate_loader_rejects_inconsistent_points(viewer, tmp_path):
    path = tmp_path / "matrix.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["from_id", "from_lat", "from_lon", "to_id", "to_lat", "to_lon"])
        writer.writerow(["A", 16, 108, "B", 16.1, 108.1])
        writer.writerow(["A", 16.2, 108, "B", 16.1, 108.1])
    with pytest.raises(ValueError, match="Inconsistent coordinates for A"):
        viewer.load_points(path)


def test_visualizer_reuses_solver_arguments(viewer):
    from solve_orienteering import build_parser, solve

    args = build_parser().parse_args(["--matrix", "data/final/od_matrix_mock.csv",
                                    "--algorithm", "vns", "--start", "H01", "--end", "H02",
                                    "--budget-min", "240", "--iterations", "1", "--seed", "42"])
    events = []
    assert solve(args) == solve(args, trace=True, progress=events.append)
    assert any(e["event"] == "operator" for e in events)
