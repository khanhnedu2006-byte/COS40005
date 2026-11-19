import csv

import pandas as pd
import numpy as np

from busroute.routing.road_dijkstra import find_route, road_instructions, road_widths


def test_dijkstra_uses_physical_distance_and_reports_width(tmp_path):
    nodes = pd.DataFrame(
        [
            (0, 16.0, 108.0),
            (1, 16.0, 108.001),
            (2, 16.001, 108.001),
            (3, 16.0, 108.002),
        ],
        columns=["node_idx", "lat", "lon"],
    )
    edges = pd.DataFrame(
        [
            (0, 1, "narrow", 100),
            (1, 3, "narrow", 100),
            (0, 2, "wide", 150),
            (2, 3, "wide", 150),
        ],
        columns=["from_idx", "to_idx", "road_name", "length_m"],
    )
    nodes.to_parquet(tmp_path / "nodes.parquet")
    edges.to_parquet(tmp_path / "edges.parquet")
    with (tmp_path / "widths.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Tên đường", "Chiều rộng mặt (m)"])
        writer.writeheader()
        writer.writerows(
            [
                {"Tên đường": "narrow", "Chiều rộng mặt (m)": "4"},
                {"Tên đường": "wide", "Chiều rộng mặt (m)": "8"},
            ]
        )
    points = [
        {"id": "A", "lat": 16.0, "lon": 108.0, "kind": "hub", "score": 0},
        {"id": "B", "lat": 16.0, "lon": 108.002, "kind": "hub", "score": 0},
        {"id": "P", "lat": 16.001, "lon": 108.001, "kind": "poi", "score": 3},
    ]
    result = find_route(
        tmp_path / "nodes.parquet",
        tmp_path / "edges.parquet",
        tmp_path / "widths.csv",
        points,
        "A",
        "B",
    )
    assert result["distance_km"] == 0.2
    assert result["collected"] == []
    assert result["score"] == 0
    assert result["narrow_m"] == 200
    assert [(step["road"], step["distance_m"]) for step in result["instructions"]] == [
        ("narrow", 200)
    ]


def test_conflicting_widths_are_unknown(tmp_path):
    file = tmp_path / "widths.csv"
    file.write_text(
        "Tên đường,Chiều rộng mặt (m)\nA,4\nA,8\nB,3.5 & 6\nC,5.5\n", encoding="utf-8"
    )
    assert road_widths(file) == {"c": 5.5}


def test_directions_group_road_edges_and_estimate_turns():
    coords = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [3.0, 1.0]]
    edges = [
        (100.0, None, "First"),
        (80.0, None, "First"),
        (60.0, None, "Second"),
        (50.0, None, "Unnamed road"),
    ]
    steps = road_instructions(coords, edges, np.array([10.0, 10.0]))
    assert [s["action"] for s in steps] == [
        "Start on",
        "Turn left onto",
        "Turn right onto",
    ]
    assert [s["distance_m"] for s in steps] == [180.0, 60.0, 50.0]
    assert steps[1]["coordinate"] == [2.0, 0.0]
    assert steps[0]["coordinates"] == coords[:3]
    assert steps[1]["coordinates"] == coords[2:4]
    assert steps[2]["coordinates"] == coords[3:]
