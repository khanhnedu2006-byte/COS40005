"""Behavioural checks for OD evaluation and heuristic solvers."""
import csv

import pytest

from busroute.routing.metaheuristics import ALNS, VNS, Algorithm
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


@pytest.mark.parametrize("solver", [ALNS, VNS])
def test_solvers_improve_and_are_reproducible(solver):
    matrix = fixture_matrix()
    algorithm: Algorithm = solver(iterations=6, seed=7)
    result = algorithm(matrix, "A", "B", 480, 180)
    assert result == algorithm(matrix, "A", "B", 480, 180)
    assert result.route[0] == "A" and result.route[-1] == "B"
    assert result.score == 14
    assert matrix.evaluate(result.route, 480, 180) == result


@pytest.mark.parametrize("solver", [ALNS, VNS])
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


@pytest.mark.parametrize("solver", [ALNS, VNS])
def test_rejects_negative_iterations(solver):
    with pytest.raises(ValueError, match="iterations"):
        solver(iterations=-1)
