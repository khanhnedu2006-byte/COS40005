"""Behavioural checks for OD evaluation and heuristic solvers."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from busroute.routing.metaheuristics import ALNS, VNS, Algorithm, AntColony
from busroute.routing.od_evaluation import shortest_route
from busroute.routing.orienteering import Leg, ODMatrix


def fixture_matrix():
    scores = {"P1": 5.0, "P2": 7.0, "P3": 2.0}
    points = {"A", "B", *scores}
    legs = {(a, b, slot): Leg(60, frozenset()) for a in points for b in points if a != b
            for slot in ("morning", "ban_noon", "afternoon", "ban_evening", "night")}
    legs[("A", "P1", "morning")] = Leg(60, frozenset({"P2"}))
    return ODMatrix(legs, scores, points)


def test_evaluator_counts_passed_and_stopped_once():
    result = fixture_matrix().evaluate(["A", "P1", "P2", "B"], 480, 180)
    assert result is not None
    assert result.score == 12
    assert result.collected == {"P1", "P2"}
    assert result.duration_s == 180


def test_evaluator_advances_slot_and_enforces_budget():
    matrix = fixture_matrix()
    del matrix.legs[("P1", "B", "ban_noon")]
    assert matrix.evaluate(["A", "P1", "B"], 629, 120) is None
    assert matrix.evaluate(["A", "P1", "B"], 480, 119) is None
    assert matrix.evaluate(["A", "P1", "B"], 480, 120) is not None


def test_result_serializes_duration_in_fractional_minutes():
    matrix = fixture_matrix()
    matrix.legs[("A", "B", "morning")] = Leg(90, frozenset())
    result = matrix.evaluate(["A", "B"], 480, 120)
    assert result.to_dict()["duration_min"] == 1.5
    assert "duration_s" not in result.to_dict()


@pytest.mark.parametrize("solver", [ALNS, VNS, AntColony])
def test_solvers_improve_and_are_reproducible(solver):
    matrix = fixture_matrix()
    algorithm: Algorithm = solver(iterations=6, seed=7)
    result = algorithm(matrix, "A", "B", 480, 180)
    assert result == algorithm(matrix, "A", "B", 480, 180)
    assert result.route[0] == "A" and result.route[-1] == "B"
    assert result.score == 14
    assert matrix.evaluate(result.route, 480, 180) == result


@pytest.mark.parametrize("solver", [ALNS, VNS, AntColony])
def test_initial_route_falls_back_to_one_stop(solver):
    matrix = fixture_matrix()
    del matrix.legs[("A", "B", "morning")]
    algorithm: Algorithm = solver(iterations=0)
    result = algorithm(matrix, "A", "B", 480, 120)
    assert len(result.route) == 3
    with pytest.raises(ValueError, match="No feasible"):
        algorithm(matrix, "A", "B", 480, 59)


def test_csv_loader_handles_bom_and_infeasible_rows(tmp_path):
    path = tmp_path / "matrix.csv"
    fields = ["from_id", "to_id", "to_type", "to_poi_score", "slot", "duration_s", "feasible", "pois_passed"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(zip(fields, ["A", "P1", "poi", 5, "morning", 60, 1, ""])))
        writer.writerow(dict(zip(fields, ["P1", "B", "hub", 0, "morning", 0, 0, ""])))
    matrix = ODMatrix.from_csv(path)
    assert matrix.evaluate(["A", "P1"], 480, 60).score == 5
    assert matrix.evaluate(["P1", "B"], 480, 60) is None


@pytest.mark.parametrize("solver", [ALNS, VNS, AntColony])
def test_rejects_negative_iterations(solver):
    with pytest.raises(ValueError, match="iterations"):
        solver(iterations=-1)


def test_shortest_route_prefers_time_over_score_and_respects_budget():
    matrix = fixture_matrix()
    matrix.legs[("A", "B", "morning")] = Leg(150, frozenset())
    result = shortest_route(matrix, "A", "B", 480, 180)
    assert result.duration_s == 120
    assert len(result.route) == 3
    assert shortest_route(matrix, "A", "B", 480, 119) is None


def test_shortest_route_can_use_intermediate_hub():
    matrix = fixture_matrix()
    matrix.points.add("C")
    matrix.legs[("A", "B", "morning")] = Leg(150, frozenset())
    matrix.legs[("A", "C", "morning")] = Leg(30, frozenset())
    matrix.legs[("C", "B", "morning")] = Leg(30, frozenset())
    assert shortest_route(matrix, "A", "B", 480, 180).route == ("A", "C", "B")


def test_dijkstra_reports_exhausted_search():
    events = []
    assert shortest_route(fixture_matrix(), "A", "B", 480, 1, progress=events.append) is None
    assert events[-1] == {"event": "search_end", "expanded": 1, "found": False, "duration_min": None}


@pytest.mark.parametrize("solver", [ALNS, VNS, AntColony])
def test_zero_iterations_reports_initial_solution(solver):
    events = []
    result = solver(iterations=0, progress=events.append)(fixture_matrix(), "A", "B", 480, 180)
    assert [e["event"] for e in events] == ["search_start", "initial_solution", "search_end"]
    assert events[-1]["iteration"] == 0 and events[-1]["best_score"] == result.score


@pytest.mark.parametrize("algorithm", ["alns", "vns", "aco", "dijkstra"])
def test_new_solver_cli_outputs_minutes(tmp_path, algorithm):
    path = tmp_path / "matrix.csv"
    fields = ["from_id", "to_id", "to_type", "to_poi_score", "slot", "duration_s", "feasible", "pois_passed"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(zip(fields, ["A", "B", "hub", 0, "morning", 90, 1, ""])))
    script = Path(__file__).resolve().parents[1] / "scripts" / "solve_orienteering.py"
    command = [sys.executable, str(script), "--matrix", str(path),
               "--algorithm", algorithm, "--start", "A", "--end", "B",
               "--budget-min", "2", "--iterations", "1"]
    quiet = subprocess.run(command, text=True, capture_output=True, check=True)
    verbose = subprocess.run(command + ["--verbose"], text=True, capture_output=True, check=True)
    result = json.loads(quiet.stdout)
    assert quiet.stderr == ""
    assert json.loads(verbose.stdout) == result
    logs = [json.loads(line) for line in verbose.stderr.splitlines()]
    assert all(entry["algorithm"] == algorithm for entry in logs)
    assert logs[0]["event"] == "search_start" and logs[-1]["event"] == "search_end"
    if algorithm == "dijkstra":
        assert logs[-1]["found"] and logs[-1]["expanded"] > 0
    else:
        assert [entry["iteration"] for entry in logs if entry["event"] == "iteration"] == [1]
        assert logs[-1]["best_score"] == result["score"]
    assert result["route"] == ["A", "B"]
    assert result["duration_min"] == 1.5
