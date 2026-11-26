"""Môi trường Gymnasium cho PPO: xe buýt đi từng cạnh trên đồ thị đường đã rút gọn.

Khái niệm (rất quan trọng khi đọc code):
  * EPISODE = một chuyến đi trọn vẹn từ POI xuất phát A tới POI đích B, gồm giờ
    khởi hành và ngân sách thời gian; xe đi cho tới khi tới B (hoặc hết ngân sách/bước).
  * STEP = một quyết định tại nút hiện tại: chọn 1 trong tối đa K cạnh đi ra.
  * Phần thưởng: + điểm POI khi lần đầu đi qua nút chứa POI, - chi phí thời gian,
    - phạt đường hẹp / chưa rõ bề rộng, + thưởng định hình tiến tới POI trung gian,
    + thưởng khi tới B đúng hạn, - phạt khi thất bại.
  * Huấn luyện theo lô episode (xem ``ppo_train.py``): chạy N episode trọn vẹn mà KHÔNG
    cập nhật mạng, rồi học MỘT LƯỢT trên toàn bộ step của N episode đó, rồi lặp lại.

Ràng buộc "phải về được B trong ngân sách" được cài vào ACTION MASK: cạnh nào khiến
không còn kịp về B (theo đường ngắn nhất tới B) bị che. Nhờ vậy mọi episode đều kết
thúc hợp lệ và agent chỉ phải học cách GOM THÊM POI trong thời gian dư trên các tuyến
đường đủ rộng cho xe 45 chỗ.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from busroute.graph.ab_estimator import GROUP_OF, PARAMS_FILE, SLOTS
from busroute.routing.road_dijkstra import road_widths
from busroute.routing.road_graph import configured_points, snap_points
from busroute.utils.geo import normalize_road_name
from busroute.utils.timeslots import slot_of

ROOT = Path(__file__).resolve().parents[3]
REDUCED_DIR = ROOT / "data" / "stage1_reduced"
WIDTHS_FILE = ROOT / "data" / "final" / "duong_pho_clean.csv"

# Ngưỡng bề rộng đường cho xe du lịch 45 chỗ (khớp với RoadGraph).
NARROW_WIDTH_M = 5.5
DEFAULT_WIDTH_M = 7.5
MAX_WIDTH_NORM_M = 15.0

# Số đặc trưng cho mỗi cạnh ứng viên và số đặc trưng toàn cục (kích thước quan sát).
EDGE_FEATURES = 9
GLOBAL_FEATURES = 8


class GraphData:
    """Dữ liệu đồ thị bất biến, dùng chung cho mọi môi trường (không copy qua mỗi bước).

    Cạnh được sắp theo nút đi (CSR): các cạnh đi ra từ nút ``u`` nằm ở
    ``indptr[u] : indptr[u+1]``. Thời gian mỗi cạnh = độ dài × giây/mét của
    (khung giờ, nhóm đường), cùng mô hình thời gian với Web/baseline.
    """

    def __init__(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        points: list[dict[str, Any]],
        params: dict[str, Any],
        widths: dict[str, float] | None = None,
    ) -> None:
        """Khởi tạo cấu trúc mảng CSR, thời gian theo khung giờ và metadata POI/bề rộng.

        Args:
            nodes: Bảng nút rút gọn có các cột ``node_idx``, ``lat``, ``lon``.
            edges: Bảng cạnh rút gọn có ``from_idx``, ``to_idx``, ``length_m``,
                ``road_class`` và tên đường (nếu có).
            points: Danh sách dictionary điểm cấu hình (POI và hub) từ
                ``configured_points()``.
            params: Tham số hiệu chỉnh tốc độ gồm khóa ``s_per_m[slot][group]``.
            widths: Ánh xạ tên đường chuẩn hóa -> bề rộng mặt đường (mét); ``None``
                nếu không có dữ liệu bề rộng.
        """
        self.n = len(nodes)
        edges = edges.sort_values(["from_idx", "to_idx"], kind="stable").reset_index(drop=True)
        self.src = edges.from_idx.to_numpy(np.int64)
        self.dst = edges.to_idx.to_numpy(np.int64)
        self.length = edges.length_m.to_numpy(np.float64)
        self.via_nodes: list[str] = (
            edges["via_nodes"].fillna("").astype(str).tolist()
            if "via_nodes" in edges.columns
            else [""] * len(edges)
        )
        self.indptr = np.r_[0, np.cumsum(np.bincount(self.src, minlength=self.n))]
        self.max_degree = int(np.diff(self.indptr).max())
        self.coords = nodes[["lon", "lat"]].to_numpy()

        # Bề rộng mặt đường và nhãn hiển thị từng cạnh (khớp quy tắc của RoadGraph).
        def width_of(name: object) -> float:
            if not widths:
                return float("nan")
            values = {
                widths[n]
                for part in str(name or "").split(" | ")
                if (n := normalize_road_name(part)) in widths
            }
            return next(iter(values)) if len(values) == 1 else float("nan")

        if "road_name" in edges.columns:
            self.width = edges.road_name.fillna("").map(width_of).to_numpy(np.float64)
        else:
            self.width = np.full(len(edges), np.nan, dtype=np.float64)
        self.narrow = np.isfinite(self.width) & (self.width < NARROW_WIDTH_M)

        self.road_label: list[str] = []
        for row in edges.to_dict("records"):
            label = next(
                (
                    row.get(key)
                    for key in ("road_name_raw", "road_ref", "road_name")
                    if isinstance(row.get(key), str) and row[key].strip()
                ),
                "Unnamed road",
            )
            self.road_label.append(label.strip().replace(" | ", " / "))

        # Thời gian (giây) của từng cạnh theo từng khung giờ.
        classes = (
            edges["road_class"]
            if "road_class" in edges.columns
            else pd.Series("residential", index=edges.index)
        )
        group = classes.map(lambda c: GROUP_OF.get(c, "local"))
        self.time = {
            s: self.length * group.map(params["s_per_m"][s]).to_numpy() for s in SLOTS
        }
        # Ma trận ngược (cạnh song song lấy cạnh nhanh nhất) để chạy Dijkstra "về" một nút.
        self._rev = {}
        for s in SLOTS:
            best = (
                pd.DataFrame({"u": self.src, "v": self.dst, "t": self.time[s]})
                .groupby(["u", "v"], as_index=False)
                .t.min()
            )
            # Đảo chiều: dijkstra từ B trên đồ thị ngược cho ra thời gian đi TỚI B.
            self._rev[s] = csr_matrix((best.t, (best.v, best.u)), shape=(self.n, self.n))
        self._to_target: dict[tuple[str, int], np.ndarray] = {}

        # POI / hub được snap vào nút của đồ thị rút gọn.
        snapped, _, _ = snap_points(nodes.reset_index(drop=True), points)
        self.point_by_id: dict[str, tuple[int, str]] = {
            p["id"]: (int(snapped[p["id"]]), str(p["name"]))
            for p in points
            if p["id"] in snapped
        }
        self.hubs = [snapped[p["id"]] for p in points if p["kind"] == "hub" and p["id"] in snapped]
        self.hub_names: dict[int, str] = {
            snapped[p["id"]]: p["name"]
            for p in points
            if p["kind"] == "hub" and p["id"] in snapped
        }
        poi = [p for p in points if p["kind"] == "poi" and p["id"] in snapped]
        self.poi_ids = [p["id"] for p in poi]
        self.poi_names = [p["name"] for p in poi]
        self.poi_by_id = {p["id"]: i for i, p in enumerate(poi)}
        self.poi_node = np.array([snapped[p["id"]] for p in poi], dtype=np.int64)
        self.poi_score = np.array([float(p["score"]) for p in poi])
        # Nút -> các chỉ số POI nằm ở nút đó.
        self.pois_at: dict[int, list[int]] = {}
        for i, node in enumerate(self.poi_node):
            self.pois_at.setdefault(int(node), []).append(i)

    @classmethod
    def load(
        cls,
        directory: Path = REDUCED_DIR,
        widths_path: Path | None = WIDTHS_FILE,
    ) -> GraphData:
        """Nạp đồ thị rút gọn, tham số tốc độ và bảng bề rộng đường từ đĩa.

        Args:
            directory: Thư mục chứa ``nodes_reduced.parquet`` và ``edges_reduced.parquet``.
            widths_path: Đường dẫn CSV bề rộng đường (``duong_pho_clean.csv``); truyền
                ``None`` hoặc đường dẫn không tồn tại để bỏ qua thông tin bề rộng.

        Returns:
            Thực thể ``GraphData`` đã sẵn sàng dùng chung cho các môi trường.

        Raises:
            OSError: Khi thiếu tệp parquet hoặc tệp tham số tốc độ.
        """
        nodes = pd.read_parquet(directory / "nodes_reduced.parquet")
        edges = pd.read_parquet(directory / "edges_reduced.parquet")
        params = json.loads(Path(PARAMS_FILE).read_text(encoding="utf-8"))
        widths = (
            road_widths(widths_path)
            if widths_path is not None and Path(widths_path).exists()
            else None
        )
        return cls(nodes, edges, configured_points(), params, widths=widths)

    def node_label(self, node: int) -> str:
        """Trả về nhãn dễ đọc của một nút (ưu tiên tên POI, rồi tên hub, cuối cùng Node #id).

        Args:
            node: Chỉ số nút (0..N-1) trong đồ thị rút gọn.

        Returns:
            Tên các POI tại nút (nối bằng `` / `` nếu trùng nút), tên hub, hoặc
            ``"Node #<node>"``.
        """
        if node in self.pois_at:
            return " / ".join(self.poi_names[i] for i in self.pois_at[node])
        if node in self.hub_names:
            return self.hub_names[node]
        return f"Node #{node}"

    def to_target(self, slot: str, target: int) -> np.ndarray:
        """Tính thời gian ngắn nhất (giây) từ MỌI nút tới ``target`` trong khung giờ ``slot``.

        Args:
            slot: Tên khung giờ (một trong ``SLOTS``).
            target: Chỉ số nút đích B trong đồ thị rút gọn.

        Returns:
            Mảng 1D độ dài ``n`` chứa thời gian đi ngắn nhất (giây) tới ``target``.
        """
        key = (slot, target)
        if key not in self._to_target:
            self._to_target[key] = dijkstra(self._rev[slot], directed=True, indices=target)
        return self._to_target[key]

    def to_nearest(self, slot: str, nodes: np.ndarray) -> np.ndarray:
        """Tính thời gian ngắn nhất (giây) từ mỗi nút tới nút gần nhất trong ``nodes``.

        Args:
            slot: Tên khung giờ (một trong ``SLOTS``).
            nodes: Mảng chỉ số các nút mục tiêu (thường là các POI chưa ghé).

        Returns:
            Mảng 1D độ dài ``n`` chứa thời gian ngắn nhất tới tập ``nodes`` (``inf`` nếu rỗng).
        """
        if len(nodes) == 0:
            return np.full(self.n, np.inf)
        return dijkstra(self._rev[slot], directed=True, indices=nodes, min_only=True)


class BusRouteEnv(gym.Env):
    """Một chuyến xe POI -> POI: mỗi step chọn một cạnh đi ra trên đồ thị đường."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        graph: GraphData,
        *,
        max_episode_steps: int = 1000,
        budget_factor: tuple[float, float] = (1.3, 2.5),
        min_extra_min: float = 15.0,
        time_penalty: float = 0.01,
        progress_coef: float = 0.02,
        revisit_penalty: float = 0.02,
        narrow_penalty: float = 1.0,
        unknown_width_penalty: float = 0.05,
        success_bonus: float = 1.0,
        fail_penalty: float = 5.0,
        safety_margin: float = 1.05,
    ) -> None:
        """Khởi tạo môi trường định tuyến từng cạnh cho xe du lịch 45 chỗ.

        Args:
            graph: Dữ liệu đồ thị rút gọn bất biến dùng chung.
            max_episode_steps: Số step tối đa mỗi episode trước khi bị cắt (truncated).
            budget_factor: Khoảng nhân ``(min, max)`` so với thời gian đi thẳng A -> B.
            min_extra_min: Số phút cộng thêm tối thiểu vào ngân sách thời gian.
            time_penalty: Điểm trừ cho mỗi phút di chuyển trên cạnh.
            progress_coef: Hệ số thưởng định hình khi tiến gần POI trung gian gần nhất.
            revisit_penalty: Điểm trừ khi quay lại nút đã đi qua trong cùng episode.
            narrow_penalty: Điểm trừ cơ sở cho mỗi km đi qua đường hẹp (< 5.5 m).
            unknown_width_penalty: Điểm trừ nhẹ cho mỗi km đi qua đường chưa rõ bề rộng.
            success_bonus: Điểm thưởng khi đến đích B trong ngân sách thời gian.
            fail_penalty: Điểm phạt khi quá hạn hoặc hết số step tối đa mà chưa tới B.
            safety_margin: Hệ số dự phòng thời gian về B trong action mask.
        """
        super().__init__()
        self.g = graph
        self.K = graph.max_degree
        self.max_steps = max_episode_steps
        self.budget_factor, self.min_extra_s = budget_factor, min_extra_min * 60
        self.time_penalty, self.progress_coef = time_penalty, progress_coef
        self.revisit_penalty = revisit_penalty
        self.narrow_penalty = narrow_penalty
        self.unknown_width_penalty = unknown_width_penalty
        self.success_bonus = success_bonus
        self.fail_penalty, self.margin = fail_penalty, safety_margin

        # Quan sát phẳng: K cạnh × EDGE_FEATURES + GLOBAL_FEATURES; hành động: chọn 1 trong K cạnh.
        size = self.K * EDGE_FEATURES + GLOBAL_FEATURES
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.K)

    # ------------------------------------------------------------------ episode
    def _resolve_endpoint(self, value: int | str) -> tuple[int, str]:
        """Chuyển mã điểm (POI 'P001' hoặc hub 'H01') hoặc chỉ số nút sang ``(node_idx, label)``.

        Args:
            value: Chuỗi mã điểm cấu hình hoặc chỉ số nút nguyên.

        Returns:
            Bộ ``(node_idx, display_name)`` tương ứng trên đồ thị rút gọn.

        Raises:
            KeyError: Nếu chuỗi mã điểm không có trong đồ thị.
        """
        if isinstance(value, str):
            return self.g.point_by_id[value]
        node = int(value)
        return node, self.g.node_label(node)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Bắt đầu episode mới (mặc định POI A -> POI B), giờ xuất phát và ngân sách thời gian.

        Args:
            seed: Seed cho bộ sinh số ngẫu nhiên của Gymnasium.
            options: Tùy chọn ghi đè gồm ``start``, ``end`` (mã điểm hoặc chỉ số nút),
                ``start_minute`` và ``budget_s``.

        Returns:
            Bộ ``(observation, info)`` ban đầu; ``info`` chứa tên và chỉ số điểm đầu/cuối
            cùng ngân sách (phút).

        Raises:
            ValueError: Nếu đồ thị không có cặp điểm đầu/cuối khả dụng.
        """
        super().reset(seed=seed)
        rng, g = self.np_random, self.g
        opt = options or {}
        if opt.get("start") is not None and opt.get("end") is not None:
            a, self.start_name = self._resolve_endpoint(opt["start"])
            b, self.end_name = self._resolve_endpoint(opt["end"])
        elif len(set(g.poi_node.tolist())) >= 2:
            si, ei = (int(x) for x in rng.choice(len(g.poi_node), size=2, replace=False))
            while g.poi_node[si] == g.poi_node[ei]:  # hai POI có thể snap vào cùng một nút
                si, ei = (int(x) for x in rng.choice(len(g.poi_node), size=2, replace=False))
            a, b = int(g.poi_node[si]), int(g.poi_node[ei])
            self.start_name, self.end_name = g.poi_names[si], g.poi_names[ei]
        else:
            # Dự phòng cho đồ thị kiểm thử nhỏ có ít hơn 2 POI phân biệt.
            pts = list(g.point_by_id.values())
            pairs = [
                (an, aname, bn, bname)
                for an, aname in pts
                for bn, bname in pts
                if an != bn and math.isfinite(float(g.to_target("morning", bn)[an]))
            ]
            if not pairs:
                raise ValueError("Không tìm thấy cặp điểm đầu/cuối liên thông trên đồ thị")
            a, self.start_name, b, self.end_name = pairs[int(rng.integers(len(pairs)))]

        self.start_minute = int(opt.get("start_minute", rng.integers(6 * 60, 20 * 60)))
        self.slot0 = slot_of(self.start_minute)
        direct = float(g.to_target(self.slot0, b)[a])
        factor = rng.uniform(*self.budget_factor)
        self.budget = float(opt.get("budget_s", direct * factor + self.min_extra_s))

        self.cur, self.target, self.origin = a, b, a
        self.elapsed, self.steps, self.score = 0.0, 0, 0.0
        self.narrow_m, self.unknown_width_m = 0.0, 0.0
        self.visited = {a}
        self.collected = np.zeros(len(g.poi_node), dtype=bool)
        # Đánh dấu POI tại điểm xuất phát đã đi qua (không cộng reward bước đầu).
        self._collect_here(a)
        self._refresh_poi_field()
        return self._observe(), {
            "start": a,
            "end": b,
            "start_name": self.start_name,
            "end_name": self.end_name,
            "budget_min": self.budget / 60,
        }

    def _slot(self) -> str:
        """Lấy tên khung giờ theo đồng hồ hiện tại của chuyến.

        Returns:
            Tên khung giờ (ví dụ ``"morning"``, ``"ban_noon"``).
        """
        return slot_of((self.start_minute + int(self.elapsed // 60)) % 1440)

    def _collect_here(self, node: int) -> float:
        """Thu điểm các POI chưa lấy ở ``node``.

        Args:
            node: Chỉ số nút hiện tại trên đồ thị rút gọn.

        Returns:
            Tổng điểm POI mới thu được tại nút này (0.0 nếu không có POI mới).
        """
        gained = 0.0
        for i in self.g.pois_at.get(node, ()):
            if not self.collected[i]:
                self.collected[i] = True
                gained += float(self.g.poi_score[i])
        self.score += gained
        return gained

    def _refresh_poi_field(self) -> None:
        """Tính lại thời gian tới POI trung gian chưa lấy gần nhất (bỏ qua đích B)."""
        left = self.g.poi_node[(~self.collected) & (self.g.poi_node != self.target)]
        self.poi_field = self.g.to_nearest(self.slot0, left)

    # ------------------------------------------------------------- observation
    def _candidates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Lấy các cạnh đi ra từ nút hiện tại và tính mask hợp lệ.

        Returns:
            Bộ 5 mảng ``(edges, time_s, to_nodes, time_to_target, valid_mask)``.
        """
        g = self.g
        lo, hi = g.indptr[self.cur], g.indptr[self.cur + 1]
        edges = np.arange(lo, hi)
        slot = self._slot()
        t = g.time[slot][edges]
        to = g.dst[edges]
        back = g.to_target(slot, self.target)[to]
        slack = self.budget - self.elapsed - t - back * self.margin
        valid = slack >= 0
        if len(edges) > 0 and not valid.any():  # không cạnh nào kịp: chỉ cho cạnh về B nhanh nhất
            valid = np.zeros(len(edges), dtype=bool)
            valid[int(np.argmin(t + back))] = True
        return edges, t, to, back, valid

    def _observe(self) -> np.ndarray:
        """Đóng gói vector quan sát phẳng gồm đặc trưng K cạnh đi ra và thông tin toàn cục.

        Returns:
            Mảng ``float32`` kích thước ``K * EDGE_FEATURES + GLOBAL_FEATURES``.
        """
        g, K = self.g, self.K
        edges, t, to, back, valid = self._candidates()
        slot = self._slot()
        here_back = float(g.to_target(slot, self.target)[self.cur])
        here_poi = float(self.poi_field[self.cur])
        n = len(edges)

        feat = np.zeros((K, EDGE_FEATURES), dtype=np.float32)
        feat[:n, 0] = valid
        feat[:n, 1] = np.minimum(g.length[edges] / 1000.0, 5.0)  # km
        feat[:n, 2] = np.minimum(t / 600.0, 5.0)  # đơn vị 10 phút
        # "Tỉ lệ tiến bộ": 1 = đi đúng đường ngắn nhất tới B; <0 = đi xa B.
        feat[:n, 3] = np.clip((here_back - back) / np.maximum(t, 1e-6), -1, 1)
        poi_to = self.poi_field[to]
        gain = np.where(
            np.isfinite(here_poi) & np.isfinite(poi_to),
            (here_poi - poi_to) / np.maximum(t, 1e-6),
            0.0,
        )
        feat[:n, 4] = np.clip(gain, -1, 1)  # tiến gần POI trung gian chưa lấy gần nhất
        for j in range(n):  # điểm POI chưa lấy ngay tại nút tới
            feat[j, 5] = sum(
                g.poi_score[i] for i in g.pois_at.get(int(to[j]), ()) if not self.collected[i]
            ) / 5.0
        feat[:n, 6] = [int(v) in self.visited for v in to]
        # Đặc trưng bề rộng mặt đường cho xe 45 chỗ: cờ đường hẹp và bề rộng chuẩn hóa.
        feat[:n, 7] = g.narrow[edges].astype(np.float32)
        w = g.width[edges]
        feat[:n, 8] = np.where(
            np.isfinite(w),
            np.clip(w / MAX_WIDTH_NORM_M, 0.0, 1.0),
            DEFAULT_WIDTH_M / MAX_WIDTH_NORM_M,
        ).astype(np.float32)

        tod = 2 * math.pi * ((self.start_minute + self.elapsed / 60) % 1440) / 1440
        remain = max(self.budget - self.elapsed, 0.0)
        glob = np.array(
            [
                remain / self.budget,
                np.clip((remain - here_back) / self.budget, 0, 1),  # thời gian dư (slack)
                np.clip(here_back / self.budget, 0, 1),
                math.sin(tod),
                math.cos(tod),
                np.clip(here_poi / self.budget, 0, 1) if np.isfinite(here_poi) else 1.0,
                self.collected.mean(),
                self.steps / self.max_steps,
            ],
            dtype=np.float32,
        )
        mask = np.zeros(K, dtype=bool)
        mask[:n] = valid
        self._mask, self._cand = mask, (edges, t, to)
        return np.concatenate([feat.ravel(), glob])

    def action_masks(self) -> np.ndarray:
        """Trả về action mask cho ``MaskablePPO``.

        Returns:
            Mảng boolean độ dài ``K`` (``True`` = khe hành động hợp lệ).
        """
        return self._mask

    def _width_reward(self, edge_idx: int) -> float:
        """Tính thành phần phạt bề rộng đường (luôn <= 0 để tránh tự quay vòng kiếm thưởng).

        Args:
            edge_idx: Chỉ số cạnh được chọn trong mảng CSR của ``GraphData``.

        Returns:
            Giá trị âm cho đường hẹp (< 5.5 m) hoặc đường chưa rõ bề rộng; ``0.0``
            khi đường đủ rộng (>= 5.5 m).
        """
        length_km = float(self.g.length[edge_idx]) / 1000.0
        w = float(self.g.width[edge_idx])
        if math.isfinite(w):
            if w < NARROW_WIDTH_M:
                # Càng hẹp hơn ngưỡng 5.5 m thì phạt trên mỗi km càng nặng.
                return -self.narrow_penalty * length_km * (1.0 + (NARROW_WIDTH_M - w) / NARROW_WIDTH_M)
            return 0.0
        return -self.unknown_width_penalty * length_km

    # -------------------------------------------------------------------- step
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Đi một cạnh và cập nhật trạng thái chuyến xe theo chuẩn Gymnasium.

        Args:
            action: Chỉ số khe cạnh đi ra (``0 <= action < degree`` và không bị che).

        Returns:
            Bộ 5 phần tử ``(obs, reward, terminated, truncated, info)`` với ``info``
            chứa chi tiết từng step (tên đường, bề rộng, phần thưởng) và tổng kết khi
            kết thúc episode.

        Raises:
            ValueError: Nếu ``action`` vượt số cạnh đi ra hoặc nằm trong action mask bị che.
        """
        action = int(action)
        if action >= len(self._cand[0]) or not self._mask[action]:
            raise ValueError(f"Hành động không hợp lệ: {action}")
        edges, t, to = self._cand
        valid_count = int(self._mask.sum())
        degree = len(edges)
        prev = self.cur
        edge_idx = int(edges[action])
        dt, nxt = float(t[action]), int(to[action])

        # Thưởng định hình (potential-based) dùng CÙNG một poi_field cho cả hai đầu cạnh.
        old = self.poi_field
        shaping = 0.0
        if np.isfinite(old[prev]) and np.isfinite(old[nxt]):
            shaping = self.progress_coef * (old[prev] - old[nxt]) / 60.0

        width_reward = self._width_reward(edge_idx)
        edge_len = float(self.g.length[edge_idx])
        edge_w = float(self.g.width[edge_idx])
        if math.isfinite(edge_w):
            if edge_w < NARROW_WIDTH_M:
                self.narrow_m += edge_len
        else:
            self.unknown_width_m += edge_len

        reward = -self.time_penalty * dt / 60.0 + shaping + width_reward
        if nxt in self.visited:
            reward -= self.revisit_penalty
        self.visited.add(nxt)
        self.elapsed += dt
        self.steps += 1
        self.cur = nxt

        new_pois = [
            self.g.poi_names[i]
            for i in self.g.pois_at.get(nxt, ())
            if not self.collected[i]
        ]
        gained = self._collect_here(nxt)
        reward += gained
        if gained > 0:
            self._refresh_poi_field()

        arrived = nxt == self.target
        late = self.elapsed > self.budget + 1e-6
        terminated = arrived or late
        truncated = (not terminated) and self.steps >= self.max_steps
        success = arrived and not late
        if success:
            reward += self.success_bonus
        elif terminated or truncated:
            reward -= self.fail_penalty

        info: dict[str, Any] = {
            "step": self.steps,
            "edge_idx": edge_idx,
            "from_node": prev,
            "to_node": nxt,
            "from_label": self.g.node_label(prev),
            "to_label": self.g.node_label(nxt),
            "road_name": self.g.road_label[edge_idx],
            "length_m": edge_len,
            "width_m": None if not math.isfinite(edge_w) else edge_w,
            "narrow": bool(self.g.narrow[edge_idx]),
            "dt_s": dt,
            "gained_score": gained,
            "new_pois": new_pois,
            "width_reward": width_reward,
            "reward": float(reward),
            "elapsed_min": self.elapsed / 60,
            "budget_min": self.budget / 60,
            "valid_actions": valid_count,
            "degree": degree,
            "action": action,
        }
        if terminated or truncated:
            info.update(
                {
                    "score": self.score,
                    "success": float(success),
                    "pois": int(self.collected.sum()),
                    "start": self.origin,
                    "end": self.target,
                    "start_name": self.start_name,
                    "end_name": self.end_name,
                    "narrow_m": self.narrow_m,
                    "unknown_width_m": self.unknown_width_m,
                }
            )
        return self._observe(), float(reward), terminated, truncated, info
