"""Graph-derived orienteering: real paths, explicit detours and unique rewards."""

import pandas as pd
import pytest

from busroute.routing.road_graph import RoadGraph, RoadProblem, RoadResult
from busroute.routing.road_metaheuristics import search


def point(id, lat, lon, score=0):
    return dict(
        id=id, name=id, lat=lat, lon=lon, score=score, kind="poi" if score else "hub"
    )


def graph_fixture(cancelled=lambda: False):
    points = [
        point("A", 16, 108),
        point("B", 16, 108.006),
        point("P", 16.003, 108.003, 5),
        point("Q", 16.006, 108.003, 7),
    ]
    nodes = pd.DataFrame(
        [(i, p["lat"], p["lon"]) for i, p in enumerate(points)],
        columns=["node_idx", "lat", "lon"],
    )
    edges = pd.DataFrame(
        [
            (0, 1, "Direct", 600),
            (0, 2, "To P", 450),
            (2, 1, "From P", 450),
            (2, 3, "To Q", 330),
            (3, 1, "From Q", 670),
            (0, 3, "To Q", 900),
        ],
        columns=["from_idx", "to_idx", "road_name", "length_m"],
    )
    params = {
        "s_per_m": {
            s: {
                g: 0.06
                for g in ["fast", "primary", "secondary", "tertiary", "minor", "local"]
            }
            for s in ["morning", "ban_noon", "afternoon", "ban_evening", "night"]
        }
    }
    return RoadGraph(
        nodes,
        edges,
        {"direct": 4, "to p": 8, "from p": 8},
        points,
        cancelled,
        time_params=params,
    )


@pytest.mark.parametrize("algorithm", ["alns", "vns", "aco"])
def test_midway_poi_is_added_even_when_dijkstra_skips_it(algorithm):
    graph = graph_fixture()
    problem = RoadProblem(graph, "A", "B", 0.95)
    calls = problem.dijkstra_calls
    assert problem.direct.score == 0
    result = search(problem, algorithm, iterations=10, seed=17)
    assert result.route == ("A", "P", "B")
    assert result.score == 5
    assert result.distance_m == 900
    assert result.quality >= problem.direct.quality
    assert search(problem, algorithm, iterations=10, seed=17) == result
    assert problem.dijkstra_calls == calls  # No graph searches during insertion trials.
    snapshot = graph.snapshot(problem, result)
    assert snapshot["collected"] == ["P"]
    assert snapshot["geometry"]["coordinates"][1] == [108.003, 16.003]
    assert sum(s["distance_m"] for s in snapshot["instructions"]) == result.distance_m
    sections = snapshot["instructions"]
    reconstructed = sections[0]["coordinates"][:1] + [
        c for step in sections for c in step["coordinates"][1:]
    ]
    assert reconstructed == snapshot["geometry"]["coordinates"]


@pytest.mark.parametrize("algorithm", ["alns", "vns", "aco"])
def test_budget_blocks_detour_and_zero_iterations_returns_greedy_seed(algorithm):
    problem = RoadProblem(graph_fixture(), "A", "B", 0.8)
    assert search(problem, algorithm, iterations=10).route == ("A", "B")
    problem = RoadProblem(graph_fixture(), "A", "B", 0.95)
    assert search(problem, algorithm, iterations=0).route == ("A", "P", "B")
    with pytest.raises(ValueError, match="non-negative"):
        search(problem, algorithm, iterations=-1)


def test_aco_can_improve_on_its_greedy_seed():
    fixture = graph_fixture()
    edges = pd.DataFrame(
        [
            (0, 1, "Direct", 600),
            (0, 2, "To P", 375),
            (2, 1, "From P", 375),
            (0, 3, "To Q", 475),
            (3, 1, "From Q", 475),
        ],
        columns=["from_idx", "to_idx", "road_name", "length_m"],
    )
    graph = RoadGraph(
        fixture.nodes,
        edges,
        {},
        list(fixture.points.values()),
        time_params=fixture.time_params,
    )
    problem = RoadProblem(graph, "A", "B", 1)

    assert search(problem, "aco", iterations=0, seed=0).route == ("A", "P", "B")
    result = search(problem, "aco", iterations=2, seed=0)
    assert result.route == ("A", "Q", "B")
    assert result.score == 7
    assert result.duration_s <= problem.budget_s


def test_unreachable_direction_and_insufficient_budget():
    graph = graph_fixture()
    with pytest.raises(ValueError, match="No directed route"):
        RoadProblem(graph, "B", "A", 1)
    with pytest.raises(ValueError, match="budget"):
        RoadProblem(graph, "A", "B", 0.5)


def test_terminal_destination_cannot_be_passed_on_the_way_to_a_poi():
    graph = graph_fixture()
    # The tempting cheap route via B must not be used as a connection to P.
    edges = pd.DataFrame(
        [(0, 1, "AB", 100), (1, 2, "BP", 100), (0, 2, "AP", 400), (2, 1, "PB", 100)],
        columns=["from_idx", "to_idx", "road_name", "length_m"],
    )
    graph = RoadGraph(
        graph.nodes,
        edges,
        {},
        list(graph.points.values()),
        time_params=graph.time_params,
    )
    problem = RoadProblem(graph, "A", "B", 0.35)
    assert problem.leg("A", "P").distance_m == 400
    assert problem.evaluate(("A", "P", "B")) is None


def test_shared_snapped_node_zero_leg_does_not_double_count_pois():
    graph = graph_fixture()
    points = list(graph.points.values()) + [
        point("P2", 16.003, 108.003, 3),
        point("Far", 17, 109, 9),
    ]
    graph = RoadGraph(
        graph.nodes, graph.edges, {}, points, time_params=graph.time_params
    )
    problem = RoadProblem(graph, "A", "B", 0.95)
    assert "Far" not in problem.candidates
    with pytest.raises(ValueError, match="300 m"):
        RoadProblem(graph, "Far", "B", 1)
    result = problem.evaluate(("A", "P", "P2", "B"))
    assert result.score == 8
    assert problem.leg("P", "P2").distance_m == 0
    assert problem.evaluate(("A", "P", "P", "B")) is None
    assert graph.snapshot(problem, result)["collected"] == ["P", "P2"]


def test_parallel_edges_choose_physical_distance_then_width_without_summing():
    graph = graph_fixture()
    edges = pd.DataFrame(
        [(99, 0, 1, "Narrow", 600), (12, 0, 1, "Wide", 600), (1, 0, 1, "Long", 650)],
        columns=["edge_idx", "from_idx", "to_idx", "road_name", "length_m"],
    )
    graph = RoadGraph(
        graph.nodes,
        edges,
        {"narrow": 4, "wide": 8, "long": 8},
        list(graph.points.values()),
        time_params=graph.time_params,
    )
    problem = RoadProblem(graph, "A", "B", 1)
    leg = problem.leg("A", "B")
    assert leg.distance_m == leg.weighted_m == 600
    assert graph.edges.iloc[leg.edges[0]].edge_idx == 12
    assert graph.snapshot(problem, problem.direct)["narrow_m"] == 0


def test_same_node_endpoints_render_valid_geometry():
    graph = graph_fixture()
    points = list(graph.points.values()) + [point("A2", 16, 108)]
    graph = RoadGraph(
        graph.nodes, graph.edges, {}, points, time_params=graph.time_params
    )
    problem = RoadProblem(graph, "A", "A2", 1)
    assert problem.direct.distance_m == 0
    coords = graph.snapshot(problem, problem.direct)["geometry"]["coordinates"]
    assert len(coords) == 2 and coords[0] == coords[1]


@pytest.mark.parametrize("algorithm", ["alns", "vns", "aco"])
def test_cancellation_in_search(algorithm):
    stop = [False]
    graph = graph_fixture(lambda: stop[0])
    problem = RoadProblem(graph, "A", "B", 1)
    stop[0] = True
    with pytest.raises(InterruptedError):
        search(problem, algorithm)


def test_score_then_width_cost_then_physical_distance_ordering():
    a = RoadResult(("A", "B"), 5, 100, 200, 1)
    b = RoadResult(("A", "P", "B"), 5, 150, 150, 1)
    c = RoadResult(("A", "B"), 6, 200, 400, 3)
    assert c.quality > b.quality > a.quality


@pytest.mark.parametrize("algorithm", ["alns", "vns", "aco"])
def test_equal_score_tour_can_prefer_longer_wide_roads(algorithm):
    graph = graph_fixture()
    points = [p.copy() for p in graph.points.values() if p["id"] != "Q"]
    next(p for p in points if p["id"] == "P")["score"] = 0
    graph = RoadGraph(
        graph.nodes,
        graph.edges,
        {"direct": 4, "to p": 8, "from p": 8},
        points,
        time_params=graph.time_params,
    )
    problem = RoadProblem(graph, "A", "B", 0.95)
    result = search(problem, algorithm, iterations=3)
    assert result.score == problem.direct.score == 0
    assert result.route == ("A", "P", "B")
    assert result.distance_m > problem.direct.distance_m
    assert result.weighted_m < problem.direct.weighted_m


def test_fastest_path_can_be_longer_than_shortest_distance():
    from busroute.graph.ab_estimator import ABEstimator

    graph = graph_fixture()
    edges = pd.DataFrame(
        [
            (0, 1, "Direct", 600, "residential"),
            (0, 2, "To P", 450, "primary"),
            (2, 1, "From P", 450, "primary"),
        ],
        columns=["from_idx", "to_idx", "road_name", "length_m", "road_class"],
    )
    params = ABEstimator.load_params()
    graph = RoadGraph(
        graph.nodes, edges, {}, list(graph.points.values()), time_params=params
    )
    problem = RoadProblem(graph, "A", "B", 3, all_pois=False)
    assert problem.direct.distance_m == 900
    assert problem.direct.duration_s == pytest.approx(
        900 * params["s_per_m"]["morning"]["primary"]
    )
    assert graph.snapshot(problem, problem.direct)["duration_min"] <= 3


def test_departure_slots_advance_between_connections_and_snapshot_matches():
    graph = graph_fixture()
    graph.times["morning"] = graph.lengths * 0.2
    graph.times["ban_noon"] = graph.lengths * 0.4
    problem = RoadProblem(graph, "A", "B", 5, start_minute=629)
    result = problem.evaluate(("A", "P", "B"))
    assert result.slots == ("morning", "ban_noon")
    assert result.duration_s == pytest.approx(270)
    assert graph.snapshot(problem, result)["duration_min"] == pytest.approx(4.5)
    problem.budget_s = 240
    assert problem.evaluate(("A", "P", "B")) is None


def test_night_to_morning_and_midnight_departures():
    graph = graph_fixture()
    graph.times["night"] = graph.lengths * 0.2
    problem = RoadProblem(graph, "A", "B", 3, start_minute=299)
    result = problem.evaluate(("A", "P", "B"))
    assert result.slots == ("night", "morning")
    assert result.duration_s == pytest.approx(117)
    problem = RoadProblem(graph, "A", "B", 4, start_minute=1439)
    assert problem.evaluate(("A", "P", "B")).slots == ("night", "night")


def test_connection_uses_an_alternate_path_when_departure_slot_path_runs_over_budget():
    fixture = graph_fixture()
    edges = pd.DataFrame(
        [
            (0, 2, "To P", 100),
            (2, 1, "From P", 80),
            (0, 3, "To Q", 120),
            (3, 1, "From Q", 90),
        ],
        columns=["from_idx", "to_idx", "road_name", "length_m"],
    )
    graph = RoadGraph(
        fixture.nodes,
        edges,
        {},
        list(fixture.points.values()),
        time_params=fixture.time_params,
    )
    graph.times["morning"][:] = [60, 10, 65, 30]
    graph.times["ban_noon"][:] = [60, 100, 65, 20]

    problem = RoadProblem(graph, "A", "B", 1.5, start_minute=629, all_pois=False)

    assert graph.advance_leg(problem.leg("A", "B", "morning"), 629, 0) == 160
    assert problem.direct.slots == ("morning",)
    assert problem.direct.connection_slots == ("ban_noon",)
    assert problem.direct.duration_s == 85
    snapshot = graph.snapshot(problem, problem.direct)
    assert snapshot["duration_min"] == pytest.approx(85 / 60)
    assert snapshot["collected"] == ["Q"]
    assert [step["road"] for step in snapshot["instructions"]] == ["To Q", "From Q"]


def test_a_faster_connection_to_a_waypoint_can_make_the_whole_tour_slower():
    fixture = graph_fixture()
    nodes = pd.DataFrame(
        [
            (0, 16, 108),
            (1, 16, 108.006),
            (2, 16.003, 108.003),
            (3, 16.001, 108.001),
            (4, 16.003, 108.001),
            (5, 16.002, 108.005),
        ],
        columns=["node_idx", "lat", "lon"],
    )
    edges = pd.DataFrame(
        [
            (0, 3, "A to X", 100),
            (3, 2, "X to P", 80),
            (0, 4, "A to Y", 120),
            (4, 2, "Y to P", 90),
            (2, 5, "P to J", 12000),
            (5, 1, "J to B", 100),
        ],
        columns=["from_idx", "to_idx", "road_name", "length_m"],
    )
    points = [fixture.points[p] for p in ("A", "B", "P")]
    graph = RoadGraph(nodes, edges, {}, points, time_params=fixture.time_params)
    graph.times["morning"][:] = [60, 10, 65, 30, 5340, 10]
    graph.times["ban_noon"][:] = [60, 100, 65, 20, 5340, 300]
    graph.times["afternoon"][:] = [60, 100, 65, 20, 5340, 10]

    problem = RoadProblem(graph, "A", "B", 92, start_minute=629)
    result = problem.evaluate(("A", "P", "B"))

    assert result is not None
    assert result.duration_s == 5510
    assert result.connection_slots[0] == "morning"
    assert graph.snapshot(problem, result)["duration_min"] == pytest.approx(5510 / 60)
