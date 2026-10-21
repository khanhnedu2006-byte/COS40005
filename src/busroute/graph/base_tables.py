"""Chặng 1 – Chuyển đồ thị OSM thành 2 bảng nền (nodes_base, edges_base) và bảng tên đường.

Người phụ trách: Nguyen Nam Khanh
Không import osmnx ở đây để có thể test bằng đồ thị giả lập.
"""
import pandas as pd

from busroute.utils.geo import normalize_road_name

# Vùng con để train từ dễ đến khó, [west, south, east, north]
# core = nội thành Đà Nẵng cũ, hoian = Hội An; node ngoài 2 vùng → "outer"
SUBGRAPH_BBOXES = {
    "core": [108.13, 15.95, 108.32, 16.13],
    "hoian": [108.27, 15.84, 108.37, 15.92],
}
NAME_SEPARATOR = " | "   # một cạnh sau simplify có thể mang nhiều tên đường


def _as_list(value):
    # OSMnx gộp thuộc tính thành list khi simplify; chuẩn hoá về list, bỏ giá trị rỗng
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if v is not None and str(v) != "nan"]
    return [] if str(value) == "nan" else [value]


def subgraph_of(lat, lon):
    # Trả về vùng con chứa toạ độ; ưu tiên theo thứ tự khai báo
    for name, (west, south, east, north) in SUBGRAPH_BBOXES.items():
        if west <= lon <= east and south <= lat <= north:
            return name
    return "outer"


def build_base_tables(G):
    # Đánh số lại node/cạnh từ 0 và lấy các cột nền của chặng 1
    node_ids = list(G.nodes)
    index_of = {n: i for i, n in enumerate(node_ids)}
    nodes = pd.DataFrame({
        "node_idx": range(len(node_ids)),
        "osm_node_id": [int(n) for n in node_ids],
        "lat": [float(G.nodes[n]["y"]) for n in node_ids],
        "lon": [float(G.nodes[n]["x"]) for n in node_ids],
        "out_degree": [int(G.out_degree(n)) for n in node_ids],
    })
    nodes["subgraph_id"] = [subgraph_of(a, b) for a, b in zip(nodes.lat, nodes.lon)]

    rows = []
    for edge_idx, (u, v, _key, data) in enumerate(G.edges(keys=True, data=True)):
        way_ids = [int(w) for w in _as_list(data.get("osmid"))]
        raw_names = [str(n) for n in _as_list(data.get("name"))]
        classes = [str(c) for c in _as_list(data.get("highway"))]
        refs = [str(r).strip() for r in _as_list(data.get("ref"))]
        # Ghép cặp tên chuẩn hoá → tên gốc; bỏ trùng theo tên chuẩn hoá để 2 cột luôn cùng số phần tử
        # (ví dụ "Đường Lê Duẩn" và "Lê Duẩn" cùng chuẩn hoá thành "lê duẩn")
        name_pairs = {}
        for raw in raw_names:
            norm = normalize_road_name(raw)
            if norm and norm not in name_pairs:
                name_pairs[norm] = raw.strip()
        rows.append({
            "edge_idx": edge_idx,
            "from_idx": index_of[u],
            "to_idx": index_of[v],
            "osm_way_id": way_ids[0] if way_ids else -1,          # way đầu tiên
            "osm_way_ids": ";".join(map(str, way_ids)),           # mọi way bị gộp (dùng cho ACCESS_SPURS)
            "road_name": NAME_SEPARATOR.join(name_pairs.keys()),
            "road_name_raw": NAME_SEPARATOR.join(name_pairs.values()),   # tên gốc để hiển thị
            "road_ref": NAME_SEPARATOR.join(dict.fromkeys(r for r in refs if r)),   # số hiệu: QL1, ĐT609…
            "road_class": classes[0] if classes else "unclassified",
            "length_m": float(data.get("length", 0.0)),
        })
    edges = pd.DataFrame(rows)
    return nodes, edges


def road_name_summary(edges):
    # Bảng tên đường duy nhất cho người làm data đối chiếu với MAIN_ROADS
    named = edges[edges.road_name != ""].copy()
    named["road_name"] = named.road_name.str.split(NAME_SEPARATOR, regex=False)
    named["road_name_raw"] = named.road_name_raw.str.split(NAME_SEPARATOR, regex=False)
    # Tách cạnh nhiều tên thành nhiều dòng; giữ cặp tên chuẩn hoá – tên gốc
    named = named.explode(["road_name", "road_name_raw"])
    summary = (named.groupby("road_name")
               .agg(road_name_raw=("road_name_raw", lambda s: s.mode().iloc[0]),
                    road_ref=("road_ref", lambda s: s[s != ""].mode().iloc[0] if (s != "").any() else ""),
                    road_class=("road_class", lambda s: s.mode().iloc[0]),
                    n_edges=("edge_idx", "count"),
                    total_length_km=("length_m", lambda s: round(s.sum() / 1000, 2)))
               .reset_index()
               .sort_values("total_length_km", ascending=False))
    return summary
