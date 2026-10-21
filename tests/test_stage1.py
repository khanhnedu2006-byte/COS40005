"""Test chặng 1 bằng đồ thị giả lập có cấu trúc giống OSMnx (không cần mạng)."""
import networkx as nx

from busroute.graph.base_tables import build_base_tables, road_name_summary, subgraph_of


def make_fake_osm_graph():
    # 4 node: 2 ở nội thành, 1 ở Hội An, 1 ở ngoài; vòng kín để liên thông mạnh
    G = nx.MultiDiGraph()
    G.add_node(101, x=108.22, y=16.06)
    G.add_node(102, x=108.23, y=16.07)
    G.add_node(103, x=108.33, y=15.88)
    G.add_node(104, x=108.00, y=15.50)
    G.add_edge(101, 102, osmid=1, name="Đường Nguyễn Văn Linh", highway="primary", length=900.0)
    G.add_edge(102, 103, osmid=[2, 3], name=["Võ Nguyên Giáp", "Trường Sa"], highway=["trunk", "primary"], length=20000.0)
    G.add_edge(103, 104, osmid=4, highway="secondary", length=45000.0)           # đường không tên
    G.add_edge(104, 101, osmid=5, name="Quốc lộ 1A", ref="QL1A", highway="trunk", length=70000.0)
    # Cạnh có 2 tên gốc khác nhau nhưng chuẩn hoá trùng nhau (lỗi gặp trên dữ liệu thật)
    G.add_edge(101, 103, osmid=[6, 7], name=["Đường Lê Duẩn", "Lê Duẩn"], highway="secondary", length=500.0)
    return G


def test_subgraph_of():
    assert subgraph_of(16.06, 108.22) == "core"
    assert subgraph_of(15.88, 108.33) == "hoian"
    assert subgraph_of(15.50, 108.00) == "outer"


def test_build_base_tables():
    nodes, edges = build_base_tables(make_fake_osm_graph())
    assert list(nodes.node_idx) == [0, 1, 2, 3]
    assert list(nodes.out_degree) == [2, 1, 1, 1]
    assert list(nodes.subgraph_id) == ["core", "core", "hoian", "outer"]
    merged = edges[edges.from_idx == 1].iloc[0]
    assert merged.osm_way_id == 2 and merged.osm_way_ids == "2;3"
    assert merged.road_name == "võ nguyên giáp | trường sa"
    assert merged.road_class == "trunk"
    assert edges[edges.from_idx == 0].iloc[0].road_name == "nguyễn văn linh"   # bỏ tiền tố "Đường"
    assert edges[edges.from_idx == 2].iloc[0].road_name == ""
    dup = edges[edges.osm_way_ids == "6;7"].iloc[0]
    assert dup.road_name == "lê duẩn" and dup.road_name_raw == "Đường Lê Duẩn"
    assert edges[edges.from_idx == 3].iloc[0].road_ref == "QL1A"


def test_road_name_summary():
    _, edges = build_base_tables(make_fake_osm_graph())
    names = road_name_summary(edges)
    assert set(names.road_name) == {"nguyễn văn linh", "võ nguyên giáp", "trường sa", "quốc lộ 1a", "lê duẩn"}
    assert names.iloc[0].road_name == "quốc lộ 1a"            # dài nhất đứng đầu
    assert names.set_index("road_name").loc["trường sa", "road_name_raw"] == "Trường Sa"
    assert names.set_index("road_name").loc["quốc lộ 1a", "road_ref"] == "QL1A"


def test_check_stage1_passes_on_fake_graph():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "check_stage1", pathlib.Path(__file__).parents[1] / "scripts" / "check_stage1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nodes, edges = build_base_tables(make_fake_osm_graph())
    results = mod.run_checks(nodes, edges, road_name_summary(edges))
    failed = [r for r in results if not r[1] and "cảnh báo" not in r[0]]
    assert not failed, failed
