"""Kiểm thử rút gọn đồ thị trên đồ thị nhỏ dựng tay."""

import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from busroute.graph.reduce import reduce_graph


def make(edges):
    """edges: danh sách (u, v, length, class). Trả về (nodes, edges) DataFrame."""
    ids = sorted({x for u, v, *_ in edges for x in (u, v)})
    nodes = pd.DataFrame({"node_idx": ids, "lat": 0.0, "lon": 0.0})
    df = pd.DataFrame(edges, columns=["from_idx", "to_idx", "length_m", "road_class"])
    df["road_name"] = "x"
    return nodes, df


def both(a, b, length, cls="primary"):
    return [(a, b, length, cls), (b, a, length, cls)]


def dist(nodes, edges):
    n = int(nodes.node_idx.max()) + 1
    # csr_matrix CỘNG dồn cạnh song song, nên phải lấy cạnh ngắn nhất trước.
    best = edges.groupby(["from_idx", "to_idx"], as_index=False).length_m.min()
    m = csr_matrix((best.length_m, (best.from_idx, best.to_idx)), shape=(n, n))
    return dijkstra(m, directed=True)


def test_dead_end_and_chain_removed_but_distance_kept():
    # Vòng 0-1-2-3 có giao lộ ở 0 và 2 (ngõ cụt 4 gắn vào 0; 5 gắn vào 2)
    edges = both(0, 1, 10) + both(1, 2, 10) + both(2, 3, 5) + both(3, 0, 5)
    edges += both(0, 4, 3) + both(2, 5, 3) + both(5, 6, 3)
    nodes, df = make(edges)
    nr, er, rep = reduce_graph(nodes, df, protected_nodes={0, 2})
    kept = set(nr.orig_node_idx)
    assert kept == {0, 2}  # 4,5,6 là ngõ cụt; 1,3 là nút đi ngang
    d = dist(nr, er)
    o2n = dict(zip(nr.orig_node_idx, nr.node_idx))
    assert d[o2n[0], o2n[2]] == 10  # min(0-1-2 = 20, 0-3-2 = 10)
    # hai cạnh song song 0<->2 còn lại, mang dấu vết nút đã gộp
    assert set(er.via_nodes) == {"1", "3"}


def test_protected_node_is_kept_even_if_dead_end():
    nodes, df = make(both(0, 1, 5) + both(1, 2, 5))
    nr, _, _ = reduce_graph(nodes, df, protected_nodes={0, 2})
    assert set(nr.orig_node_idx) == {0, 2}  # nút 1 đi ngang bị gộp, 0 và 2 được giữ


def test_one_way_chain_keeps_direction():
    # 0 -> 1 -> 2 chỉ một chiều; 2 -> 0 đường về riêng
    edges = [(0, 1, 4, "primary"), (1, 2, 6, "primary"), (2, 0, 7, "primary")]
    nodes, df = make(edges)
    nr, er, _ = reduce_graph(nodes, df, protected_nodes={0, 2})
    assert len(er) == 2
    d = dist(nr, er)
    o2n = dict(zip(nr.orig_node_idx, nr.node_idx))
    assert d[o2n[0], o2n[2]] == 10
    assert d[o2n[2], o2n[0]] == 7


def test_strict_mode_keeps_class_change():
    edges = both(0, 1, 5, "primary") + both(1, 2, 5, "residential")
    nodes, df = make(edges)
    strict, _, _ = reduce_graph(nodes, df, {0, 2}, strict=True)
    relaxed, _, _ = reduce_graph(nodes, df, {0, 2}, strict=False)
    assert 1 in set(strict.orig_node_idx)
    assert 1 not in set(relaxed.orig_node_idx)
