"""Kiểm thử module trực quan hóa PPO (khôi phục via_nodes, rollout trace, HTML và PNG)."""

import numpy as np
import pytest

from busroute.routing.ppo_env import REDUCED_DIR, GraphData
from busroute.viz.ppo_viz import (
    build_html_visualization,
    expand_edge_coords,
    load_orig_coords,
    rollout_trace,
    save_static_figure,
)

pytestmark = pytest.mark.skipif(
    not (REDUCED_DIR / "nodes_reduced.parquet").exists(),
    reason="chạy scripts/reduce_graph.py trước",
)


@pytest.fixture(scope="module")
def graph():
    """Nạp một lần ``GraphData`` cho toàn bộ module kiểm thử."""
    return GraphData.load()


def test_expand_edge_coords_handles_contracted_edge(graph):
    """Cạnh đã gộp có ``via_nodes`` phải bung ra > 2 điểm nối đúng hai đầu cạnh."""
    orig_coords = load_orig_coords()
    if orig_coords is None:
        pytest.skip("thiếu data/stage1/nodes_base.parquet")
    contracted = next(i for i, v in enumerate(graph.via_nodes) if v)
    pts = expand_edge_coords(graph, contracted, orig_coords)
    u, v = int(graph.src[contracted]), int(graph.dst[contracted])

    assert len(pts) > 2
    assert np.allclose(pts[0], [graph.coords[u, 1], graph.coords[u, 0]])
    assert np.allclose(pts[-1], [graph.coords[v, 1], graph.coords[v, 0]])


def test_rollout_trace_and_outputs(graph, tmp_path):
    """Xuất HTML tương tác và PNG tĩnh đều thành công và chứa đúng tên POI."""
    orig_coords = load_orig_coords()
    greedy = rollout_trace(
        graph, "greedy", seed=0, orig_coords=orig_coords, max_episode_steps=40
    )
    shortest = rollout_trace(
        graph, "shortest", seed=0, orig_coords=orig_coords, max_episode_steps=40
    )
    assert greedy["start_name"] == shortest["start_name"]
    assert greedy["end_name"] == shortest["end_name"]
    assert len(greedy["steps"]) >= 1

    traces = {"greedy": greedy, "shortest": shortest}
    out_html = build_html_visualization(traces, graph, tmp_path / "viz.html")
    out_png = save_static_figure(traces, graph, tmp_path / "viz.png")

    html_text = out_html.read_text(encoding="utf-8")
    assert greedy["start_name"] in html_text
    assert greedy["end_name"] in html_text
    assert "step-slider" in html_text
    assert "ppo-payload" in html_text
    assert "maplibre-gl" in html_text
    assert "maplibregl.Map" in html_text
    assert "route-progress" in html_text
    assert "leaflet" not in html_text.lower()
    assert out_png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
