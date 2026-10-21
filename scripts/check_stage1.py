"""Kiểm tra đầu ra chặng 1 trước khi giao cho chặng 2. Thoát với mã 1 nếu có lỗi.

Người phụ trách: Nguyen Nam Khanh
Chạy: python scripts/check_stage1.py
"""
import argparse
import os
import sys

import networkx as nx
import pandas as pd

DA_NANG_BBOX = [107.20, 14.95, 108.75, 16.25]   # [west, south, east, north]
NODE_COLUMNS = ["node_idx", "osm_node_id", "lat", "lon", "out_degree", "subgraph_id"]
EDGE_COLUMNS = ["edge_idx", "from_idx", "to_idx", "osm_way_id", "osm_way_ids",
                "road_name", "road_name_raw", "road_ref", "road_class", "length_m"]
MAJOR_CLASSES = ["motorway", "trunk", "primary", "secondary"]   # đường xe 45 chỗ chủ yếu đi


def run_checks(nodes, edges, names):
    # Trả về danh sách (tên kiểm tra, đạt?, chi tiết)
    results = []
    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    check("Đủ cột nodes", set(NODE_COLUMNS) <= set(nodes.columns), str(set(NODE_COLUMNS) - set(nodes.columns)))
    check("Đủ cột edges", set(EDGE_COLUMNS) <= set(edges.columns), str(set(EDGE_COLUMNS) - set(edges.columns)))
    check("node_idx liên tục 0..N-1, không trùng",
          (nodes.node_idx.sort_values().values == range(len(nodes))).all())
    check("edge_idx liên tục 0..E-1, không trùng",
          (edges.edge_idx.sort_values().values == range(len(edges))).all())
    check("from_idx/to_idx hợp lệ",
          edges.from_idx.between(0, len(nodes) - 1).all() and edges.to_idx.between(0, len(nodes) - 1).all())
    check("Không có giá trị rỗng ở cột bắt buộc",
          not nodes[NODE_COLUMNS].isna().any().any()
          and not edges[["edge_idx", "from_idx", "to_idx", "osm_way_id", "road_class", "length_m"]].isna().any().any())
    bad_len = int((edges.length_m <= 0).sum())
    check("length_m > 0", bad_len == 0, f"{bad_len} cạnh lỗi")
    west, south, east, north = DA_NANG_BBOX
    outside = int((~nodes.lon.between(west, east) | ~nodes.lat.between(south, north)).sum())
    check("Toạ độ trong bbox Đà Nẵng", outside == 0, f"{outside} node ngoài bbox")

    G = nx.DiGraph()
    G.add_nodes_from(nodes.node_idx)
    G.add_edges_from(zip(edges.from_idx, edges.to_idx))
    n_scc = nx.number_strongly_connected_components(G)
    check("Đồ thị liên thông mạnh (1 thành phần)", n_scc == 1, f"{n_scc} thành phần")

    check("Có đủ 3 vùng core / hoian / outer",
          {"core", "hoian"} <= set(nodes.subgraph_id), str(nodes.subgraph_id.value_counts().to_dict()))
    check("Danh sách tên đường không rỗng", len(names) > 0, f"{len(names)} tên")
    # Đường dân sinh không tên là bình thường; chỉ đường lớn cần có tên hoặc số hiệu để lập whitelist
    major = edges[edges.road_class.isin(MAJOR_CLASSES)]
    no_id_pct = 100 * ((major.road_name == "") & (major.road_ref == "")).mean()
    check("Đường lớn không tên và không số hiệu < 10 %", no_id_pct < 10, f"{no_id_pct:.1f} %")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/stage1")
    ap.add_argument("--names", default="reports/osm_road_names.csv")
    args = ap.parse_args()
    nodes = pd.read_parquet(os.path.join(args.dir, "nodes_base.parquet"))
    edges = pd.read_parquet(os.path.join(args.dir, "edges_base.parquet"))
    names = pd.read_csv(args.names, encoding="utf-8-sig")

    results = run_checks(nodes, edges, names)
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" – {detail}" if detail and not ok else ""))
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} kiểm tra đạt.")
    print("→ Sẵn sàng giao chặng 2." if not failed else "→ Chưa được giao, sửa các mục FAIL.")
    sys.exit(1 if failed else 0)
