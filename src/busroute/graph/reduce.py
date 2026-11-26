"""Rút gọn đồ thị đường bộ mà vẫn giữ nguyên cấu trúc (khoảng cách đường ngắn nhất).

Hai bước, đều KHÔNG làm đổi khoảng cách ngắn nhất giữa các nút còn lại:

1. ``prune_dead_ends``: xoá lặp lại các nút "cụt" (ngõ cụt, chỉ có 1 láng giềng).
   Một nút cụt không thể nằm giữa đường ngắn nhất của hai nút khác, vì đi vào
   rồi phải quay ra đúng lối cũ.
2. ``contract_chains``: nút đi ngang (đúng 2 láng giềng a và b) được gộp lại:
   ``a -> x -> b`` thành một cạnh ``a -> b`` có độ dài bằng tổng hai cạnh.
   Giao lộ (>= 3 láng giềng) và khung đường chính được giữ nguyên.

Các nút được bảo vệ (điểm POI/hub đã snap vào đồ thị) không bao giờ bị xoá.
Đồ thị gốc không bị sửa; hàm trả về bảng mới với id được đánh số lại 0..N-1.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

# Các cột thuộc tính của cạnh được giữ lại (nếu bảng gốc có).
_ATTRS = (
    "osm_way_id",
    "osm_way_ids",
    "road_name",
    "road_name_raw",
    "road_ref",
    "road_class",
)


class _Graph:
    """Đồ thị có hướng dạng danh sách kề, cho phép cạnh song song.

    ``out[u][v]`` là danh sách các cạnh u->v; ``inc[v][u]`` trỏ tới cùng danh sách,
    nhờ vậy duyệt được cả hai chiều với chi phí O(bậc).
    """

    def __init__(self, edges: pd.DataFrame):
        """Nạp danh sách cạnh có hướng vào bảng kề hai chiều.

        Args:
            edges: DataFrame cạnh gốc có tối thiểu ``from_idx``, ``to_idx``, ``length_m``.
        """
        self.out: dict[int, dict[int, list[dict]]] = defaultdict(dict)
        self.inc: dict[int, dict[int, list[dict]]] = defaultdict(dict)
        keep = [c for c in _ATTRS if c in edges.columns]
        for row in edges[["from_idx", "to_idx", "length_m", *keep]].to_dict("records"):
            u, v = int(row.pop("from_idx")), int(row.pop("to_idx"))
            if u == v:  # bỏ khuyên (self-loop), vô dụng cho việc đi tới đích
                continue
            row["via"] = []  # các nút gốc đã bị gộp nằm giữa u và v (theo thứ tự đi)
            row["n_orig"] = 1
            self.add(u, v, row)

    def add(self, u: int, v: int, edge: dict) -> None:
        """Thêm một cạnh có hướng ``u -> v`` vào cả hai bảng kề ra và vào.

        Args:
            u: Chỉ số nút xuất phát.
            v: Chỉ số nút đích.
            edge: Dictionary thuộc tính cạnh (``length_m``, ``via``, ``n_orig``, ...).
        """
        lst = self.out[u].setdefault(v, [])
        lst.append(edge)
        self.inc[v][u] = lst

    def neighbours(self, x: int) -> set[int]:
        """Lấy tập láng giềng vô hướng của nút ``x`` (không tính chính nó).

        Args:
            x: Chỉ số nút cần xét.

        Returns:
            Tập chỉ số các nút có cạnh nối với ``x`` theo ít nhất một chiều.
        """
        return (set(self.out.get(x, ())) | set(self.inc.get(x, ()))) - {x}

    def remove_node(self, x: int) -> set[int]:
        """Xoá nút ``x`` cùng mọi cạnh liên quan khỏi đồ thị.

        Args:
            x: Chỉ số nút cần xoá.

        Returns:
            Tập chỉ số các láng giềng cũ của ``x`` trước khi xoá.
        """
        around = self.neighbours(x)
        for v in list(self.out.get(x, ())):
            self.inc[v].pop(x, None)
        for u in list(self.inc.get(x, ())):
            self.out[u].pop(x, None)
        self.out.pop(x, None)
        self.inc.pop(x, None)
        return around


def prune_dead_ends(g: _Graph, nodes: set[int], protected: set[int]) -> int:
    """Xoá lặp các nút cụt (bậc vô hướng <= 1) chưa được bảo vệ.

    Args:
        g: Đồ thị danh sách kề đang xử lý (bị biến đổi tại chỗ).
        nodes: Tập nút còn sống trong đồ thị (bị biến đổi tại chỗ).
        protected: Tập chỉ số nút được bảo vệ, không bao giờ bị xoá.

    Returns:
        Tổng số nút cụt đã bị xoá.
    """
    stack = [n for n in nodes if len(g.neighbours(n)) <= 1 and n not in protected]
    removed = 0
    while stack:
        n = stack.pop()
        if n not in nodes or n in protected or len(g.neighbours(n)) > 1:
            continue
        nodes.discard(n)
        removed += 1
        # Xoá n có thể biến láng giềng của nó thành nút cụt mới.
        for m in g.remove_node(n):
            if m in nodes and m not in protected and len(g.neighbours(m)) <= 1:
                stack.append(m)
    return removed


def _same_attrs(e1: dict, e2: dict) -> bool:
    """Kiểm tra hai cạnh nối tiếp có cùng loại đường và cùng tên đường hay không.

    Args:
        e1: Dictionary thuộc tính cạnh thứ nhất.
        e2: Dictionary thuộc tính cạnh thứ hai.

    Returns:
        ``True`` nếu ``road_class`` và ``road_name`` trùng nhau.
    """
    return e1.get("road_class") == e2.get("road_class") and (
        e1.get("road_name") == e2.get("road_name")
    )


def _merge(e1: dict, e2: dict, mid: int) -> dict:
    """Gộp hai cạnh nối tiếp ``e1`` (``a->mid``) và ``e2`` (``mid->b``) thành ``a->b``.

    Args:
        e1: Cạnh đi vào nút ``mid``.
        e2: Cạnh đi ra từ nút ``mid``.
        mid: Chỉ số nút trung gian bị rút gọn.

    Returns:
        Dictionary cạnh mới có ``length_m`` bằng tổng và ``via`` lưu thứ tự nút đã gộp.
    """
    # Thuộc tính lấy theo đoạn dài hơn (khi nghiêm ngặt thì hai đoạn giống nhau).
    main = e1 if e1["length_m"] >= e2["length_m"] else e2
    merged = {k: v for k, v in main.items() if k not in ("via", "n_orig", "length_m")}
    merged["length_m"] = e1["length_m"] + e2["length_m"]
    merged["via"] = e1["via"] + [mid] + e2["via"]
    merged["n_orig"] = e1["n_orig"] + e2["n_orig"]
    return merged


def contract_chains(
    g: _Graph, nodes: set[int], protected: set[int], strict: bool = True
) -> int:
    """Gộp các nút đi ngang (đúng 2 láng giềng) để giữ nguyên cấu trúc đường ngắn nhất.

    Một nút ``x`` được gộp khi:
      * không được bảo vệ và có đúng 2 láng giềng ``a, b``;
      * mỗi cặp ``(a,x)``, ``(x,b)``, ``(b,x)``, ``(x,a)`` có tối đa 1 cạnh (không song song);
      * hướng đi nhất quán: ``a->x`` có thì ``x->b`` phải có (và ngược lại ``b->x`` với ``x->a``),
        để không tạo ra hay làm mất đường một chiều;
      * ``strict``: hai cạnh được gộp cùng ``road_class`` và ``road_name``.

    Args:
        g: Đồ thị danh sách kề đang xử lý (bị biến đổi tại chỗ).
        nodes: Tập nút còn sống trong đồ thị (bị biến đổi tại chỗ).
        protected: Tập nút được bảo vệ (POI/hub), không bị gộp.
        strict: Nếu ``True``, chỉ gộp khi hai đoạn có cùng loại đường và tên đường.

    Returns:
        Số nút trung gian đã được gộp thành công.
    """
    queue = [n for n in nodes if n not in protected]
    merged_count = 0
    while queue:
        x = queue.pop()
        if x not in nodes or x in protected:
            continue
        around = g.neighbours(x)
        if len(around) != 2:
            continue
        a, b = sorted(around)
        pairs = []  # các phép đi xuyên qua x: (đầu, cuối)
        ok = True
        for p, q in ((a, b), (b, a)):
            into = g.out.get(p, {}).get(x, [])
            outof = g.out.get(x, {}).get(q, [])
            if len(into) > 1 or len(outof) > 1 or bool(into) != bool(outof):
                ok = False
                break
            if into:
                if strict and not _same_attrs(into[0], outof[0]):
                    ok = False
                    break
                pairs.append((p, q, into[0], outof[0]))
        # Các cạnh lệch hướng (chỉ có a->x mà không có x->b) -> không gộp.
        if not ok or not pairs:
            continue
        new_edges = [(p, q, _merge(e1, e2, x)) for p, q, e1, e2 in pairs]
        g.remove_node(x)
        nodes.discard(x)
        for p, q, edge in new_edges:
            g.add(p, q, edge)
        merged_count += 1
        queue.extend(n for n in (a, b) if n not in protected)
    return merged_count


def reduce_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    protected_nodes: set[int],
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Rút gọn đồ thị và trả về ``(nodes_reduced, edges_reduced, report)``.

    Args:
        nodes: Bảng nút gốc (cột ``node_idx``, ``lat``, ``lon``, ...).
        edges: Bảng cạnh gốc (cột ``from_idx``, ``to_idx``, ``length_m``, ...).
        protected_nodes: Tập ``node_idx`` gốc không được xoá (nút POI/hub).
        strict: ``True`` thì chỉ gộp các đoạn cùng loại đường và cùng tên.

    Returns:
        Bộ ``(nodes_reduced, edges_reduced, report)`` với id nút đánh số lại
        ``0..N-1``, giữ dấu vết qua ``orig_node_idx`` và ``via_nodes``.

    Raises:
        ValueError: Nếu có nút trong ``protected_nodes`` không tồn tại trong ``nodes``.
    """
    g = _Graph(edges)
    alive = set(int(n) for n in nodes.node_idx)
    protected = set(int(n) for n in protected_nodes)
    missing = protected - alive
    if missing:
        raise ValueError(f"Nút được bảo vệ không có trong đồ thị: {sorted(missing)[:5]}")

    report = {"nodes_before": len(alive), "edges_before": len(edges)}
    report["dead_ends_removed"] = prune_dead_ends(g, alive, protected)
    report["after_prune"] = len(alive)
    report["chains_contracted"] = contract_chains(g, alive, protected, strict)
    # Gộp xong có thể sinh thêm nút cụt (hiếm); quét lại cho chắc.
    report["dead_ends_removed"] += prune_dead_ends(g, alive, protected)

    # Đánh số lại các nút còn sống theo thứ tự gốc.
    kept = nodes[nodes.node_idx.isin(alive)].sort_values("node_idx").reset_index(drop=True)
    new_id = {int(o): i for i, o in enumerate(kept.node_idx)}
    rows = []
    for u, targets in g.out.items():
        if u not in alive:
            continue
        for v, lst in targets.items():
            if v not in alive:
                continue
            for e in lst:
                rows.append(
                    {
                        "from_idx": new_id[u],
                        "to_idx": new_id[v],
                        **{k: e.get(k) for k in _ATTRS if k in e},
                        "length_m": e["length_m"],
                        "n_orig_edges": e["n_orig"],
                        "via_nodes": ";".join(map(str, e["via"])),
                    }
                )
    edges_r = pd.DataFrame(rows).sort_values(["from_idx", "to_idx"], kind="stable")
    edges_r = edges_r.reset_index(drop=True)
    edges_r.insert(0, "edge_idx", edges_r.index)

    nodes_r = kept.rename(columns={"node_idx": "orig_node_idx"})
    nodes_r.insert(0, "node_idx", range(len(nodes_r)))
    nodes_r["is_protected"] = nodes_r.orig_node_idx.isin(protected)
    if "out_degree" in nodes_r:  # tính lại bậc ra theo đồ thị mới
        nodes_r["out_degree"] = (
            edges_r.groupby("from_idx").size().reindex(nodes_r.node_idx, fill_value=0).to_numpy()
        )
    report.update(
        nodes_after=len(nodes_r),
        edges_after=len(edges_r),
        node_ratio=len(nodes_r) / report["nodes_before"],
    )
    return nodes_r, edges_r, report
