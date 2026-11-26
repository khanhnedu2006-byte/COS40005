"""Rút gọn đồ thị stage1 (~50% số nút) và kiểm tra khoảng cách được bảo toàn.

Chạy:
    python scripts/reduce_graph.py                 # gộp 'nghiêm ngặt' (cùng loại + tên đường)
    python scripts/reduce_graph.py --relaxed       # gộp mọi nút đi ngang
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from busroute.graph.reduce import reduce_graph  # noqa: E402
from busroute.routing.road_graph import configured_points, snap_points  # noqa: E402


def to_matrix(n_nodes: int, edges: pd.DataFrame) -> csr_matrix:
    """Chuyển bảng cạnh thành ma trận kề thưa theo độ dài (cạnh song song lấy cạnh ngắn nhất).

    Args:
        n_nodes: Tổng số nút trong đồ thị.
        edges: DataFrame cạnh gồm ``from_idx``, ``to_idx`` và ``length_m``.

    Returns:
        Ma trận ``csr_matrix`` kích thước ``(n_nodes, n_nodes)`` chứa độ dài cạnh nhỏ nhất.
    """
    best = edges.groupby(["from_idx", "to_idx"], as_index=False).length_m.min()
    return csr_matrix((best.length_m, (best.from_idx, best.to_idx)), shape=(n_nodes,) * 2)


def check_distances(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    nodes_r: pd.DataFrame,
    edges_r: pd.DataFrame,
    pairs: int = 150,
    seed: int = 0,
) -> tuple[float, bool]:
    """Kiểm tra khoảng cách đường ngắn nhất trước và sau khi rút gọn trên một mẫu nút.

    Args:
        nodes: Bảng nút đồ thị gốc.
        edges: Bảng cạnh đồ thị gốc.
        nodes_r: Bảng nút sau rút gọn (có cột ``orig_node_idx``).
        edges_r: Bảng cạnh sau rút gọn.
        pairs: Số nút xuất phát lấy mẫu ngẫu nhiên để chạy Dijkstra.
        seed: Seed chọn mẫu nút xuất phát.

    Returns:
        Bộ ``(max_error_m, reachability_identical)`` biểu thị sai số độ dài lớn nhất
        và việc có giữ nguyên tính liên thông tới được hay không.
    """
    rng = np.random.default_rng(seed)
    orig = to_matrix(len(nodes), edges)
    red = to_matrix(len(nodes_r), edges_r)
    # đồ thị gốc đánh số node_idx = 0..N-1 liên tục
    src = rng.choice(len(nodes_r), size=min(pairs, len(nodes_r)), replace=False)
    d_red = dijkstra(red, directed=True, indices=src)
    d_org = dijkstra(orig, directed=True, indices=nodes_r.orig_node_idx.to_numpy()[src])
    d_org = d_org[:, nodes_r.orig_node_idx.to_numpy()]
    finite = np.isfinite(d_org)
    diff = np.abs(d_org[finite] - d_red[finite])
    return float(diff.max()), bool((finite == np.isfinite(d_red)).all())


def main() -> None:
    """Đọc đồ thị stage1, chạy rút gọn, kiểm tra bảo toàn khoảng cách và ghi parquet."""
    ap = argparse.ArgumentParser(
        description="Rút gọn đồ thị đường bộ stage1 và kiểm chứng bảo toàn khoảng cách."
    )
    ap.add_argument("--relaxed", action="store_true", help="Gộp cả khi đổi loại hoặc tên đường.")
    ap.add_argument(
        "--out",
        default=str(ROOT / "data/stage1_reduced"),
        help="Thư mục đầu ra lưu nodes_reduced.parquet và edges_reduced.parquet.",
    )
    args = ap.parse_args()

    nodes = pd.read_parquet(ROOT / "data/stage1/nodes_base.parquet")
    edges = pd.read_parquet(ROOT / "data/stage1/edges_base.parquet")

    # Nút được bảo vệ = nút đồ thị mà 50 POI + 5 hub snap vào.
    snapped, _, _ = snap_points(nodes, configured_points())
    protected = {int(nodes.node_idx.iloc[i]) for i in snapped.values()}

    nodes_r, edges_r, rep = reduce_graph(nodes, edges, protected, strict=not args.relaxed)
    for k, v in rep.items():
        print(f"{k:>20}: {v:.3f}" if isinstance(v, float) else f"{k:>20}: {v}")

    n_comp, _ = connected_components(to_matrix(len(nodes_r), edges_r), connection="strong")
    max_err, same_reach = check_distances(nodes, edges, nodes_r, edges_r)
    print(f"{'strong components':>20}: {n_comp}")
    print(f"{'max distance error':>20}: {max_err:.6f} m  (reachability identical: {same_reach})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nodes_r.to_parquet(out / "nodes_reduced.parquet", index=False)
    edges_r.to_parquet(out / "edges_reduced.parquet", index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
