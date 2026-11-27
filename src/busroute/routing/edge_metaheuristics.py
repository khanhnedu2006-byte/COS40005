"""Các thuật toán định tuyến ở mức cạnh (Dijkstra, ALNS, VNS, ACO) trên đồ thị đường.

Mọi bộ giải ở mức cạnh đều kế thừa ``StreetBaseSolver(BaseRouteSolver)`` với hợp đồng
đầu vào ``RouteRequest`` / ``RoadDatasetSpec`` và đầu ra chuẩn hóa ``RouteOutput``.
"""

from __future__ import annotations

import heapq
import json
import math
import multiprocessing as mp
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from busroute.graph.ab_estimator import PARAMS_FILE, SLOTS
from busroute.routing.base import (
    BaseRouteSolver,
    RoadDatasetSpec,
    RouteOutput,
    RouteRequest,
    _fork_worker_dispatch,
)
from busroute.routing.ppo_env import NARROW_WIDTH_M, REDUCED_DIR, GraphData
from busroute.routing.road_dijkstra import road_instructions, road_widths
from busroute.routing.road_graph import NoRoadRouteError, resolve_road_tables
from busroute.utils.timeslots import slot_of

# Hằng số phần thưởng mặc định của BusRouteEnv dùng đồng nhất cho ACO ở mức cạnh.
PPO_TIME_PENALTY = 0.01
PPO_PROGRESS_COEF = 0.02
PPO_REVISIT_PENALTY = 0.02
PPO_NARROW_PENALTY = 1.0
PPO_UNKNOWN_WIDTH_PENALTY = 0.05
PPO_SUCCESS_BONUS = 1.0


@dataclass(frozen=True)
class EdgeWalk:
    """Một hành trình hợp lệ nối các cạnh liên tiếp từ nút đầu tới nút đích."""

    edges: tuple[int, ...]
    nodes: tuple[int, ...]
    duration_s: float
    distance_m: float
    weighted_m: float
    narrow_m: float
    unknown_width_m: float
    collected_mask: int
    score: float

    @property
    def quality(self) -> tuple[float, float, float, float]:
        """Thước đo chất lượng đồng nhất với ``RoadResult``: điểm cao nhất, ưu tiên đường rộng."""
        return (
            self.score,
            -round(self.weighted_m, 6),
            -round(self.duration_s, 6),
            -round(self.distance_m, 6),
        )


@dataclass(frozen=True)
class StreetContext:
    """Ngữ cảnh thực thi đã nạp cho nhóm bộ giải ở mức cạnh."""

    graph: GraphData
    orig_coords: np.ndarray | None = None


class StreetBaseSolver(BaseRouteSolver[StreetContext, EdgeWalk]):
    """Lớp cơ sở cho các bộ giải định tuyến từng cạnh trên ``GraphData``."""

    mode: ClassVar[str] = "street"
    _fork_state: ClassVar[dict[str, Any]] = {}

    def __init__(self, *, max_steps: int = 1000) -> None:
        self.max_steps = max_steps

    def parse_data(self, data: RoadDatasetSpec, request: RouteRequest) -> StreetContext:
        """Nạp bảng nút/cạnh (hoặc đồ thị rút gọn), bề rộng đường và khởi tạo ``GraphData``."""
        request.check_cancelled()
        nodes_df, edges_df, orig_coords = resolve_road_tables(
            data.nodes_path,
            data.edges_path,
            reduced_dir=data.reduced_dir,
            use_reduced=data.use_reduced,
        )
        widths = road_widths(data.widths_path)
        params = json.loads(Path(PARAMS_FILE).read_text(encoding="utf-8"))
        graph = GraphData(nodes_df, edges_df, list(data.points), params, widths=widths)
        return StreetContext(graph=graph, orig_coords=orig_coords)

    def format_output(
        self, context: StreetContext, result: EdgeWalk, request: RouteRequest
    ) -> RouteOutput:
        """Chuyển ``EdgeWalk`` sang ``RouteOutput`` chuẩn hóa kèm hình học và chỉ dẫn."""
        snap = self._walk_to_snapshot(
            context.graph,
            result,
            request.start,
            request.end,
            orig_coords=context.orig_coords,
        )
        return RouteOutput.from_snapshot(snap)

    @staticmethod
    def _node_poi_mask(graph: GraphData, node: int) -> int:
        mask = 0
        for idx in graph.pois_at.get(node, ()):
            mask |= 1 << idx
        return mask

    @staticmethod
    def _mask_score(graph: GraphData, mask: int) -> float:
        return sum(
            float(graph.poi_score[idx])
            for idx in range(len(graph.poi_score))
            if mask & (1 << idx)
        )

    @staticmethod
    def _prewarm_graph(graph: GraphData, end_node: int, start_minute: int) -> None:
        """Tiền tính bảng khoảng cách đích và bảng POI một lần trước khi fork."""
        for slot in SLOTS:
            graph.to_target(slot, end_node)
        graph.poi_distance_table(slot_of(start_minute))

    @classmethod
    def _evaluate_walk(
        cls,
        graph: GraphData,
        start_node: int,
        end_node: int,
        edges: Sequence[int],
        start_minute: int,
        budget_s: float,
    ) -> EdgeWalk | None:
        """Kiểm tra tính hợp lệ và tính toàn bộ chỉ số cho chuỗi cạnh trên ``GraphData``."""
        cur = start_node
        elapsed = 0.0
        distance_m = 0.0
        weighted_m = 0.0
        narrow_m = 0.0
        unknown_width_m = 0.0
        collected_mask = cls._node_poi_mask(graph, start_node)
        nodes = [start_node]

        for raw_edge in edges:
            edge = int(raw_edge)
            if edge < 0 or edge >= len(graph.dst) or int(graph.src[edge]) != cur:
                return None
            slot = slot_of((start_minute + int(elapsed // 60)) % 1440)
            elapsed += float(graph.time[slot][edge])
            if elapsed > budget_s + 1e-6:
                return None
            length = float(graph.length[edge])
            width = float(graph.width[edge])
            is_narrow = bool(graph.narrow[edge])
            distance_m += length
            weighted_m += length * (2.0 if is_narrow else 1.0)
            if math.isfinite(width):
                if is_narrow:
                    narrow_m += length
            else:
                unknown_width_m += length
            cur = int(graph.dst[edge])
            nodes.append(cur)
            collected_mask |= cls._node_poi_mask(graph, cur)

        if cur != end_node or elapsed > budget_s + 1e-6:
            return None
        return EdgeWalk(
            edges=tuple(int(e) for e in edges),
            nodes=tuple(nodes),
            duration_s=elapsed,
            distance_m=distance_m,
            weighted_m=weighted_m,
            narrow_m=narrow_m,
            unknown_width_m=unknown_width_m,
            collected_mask=collected_mask,
            score=cls._mask_score(graph, collected_mask),
        )

    @staticmethod
    def _walk_to_snapshot(
        graph: GraphData,
        walk: EdgeWalk,
        start: str,
        end: str,
        *,
        orig_coords: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Chuyển ``EdgeWalk`` sang dictionary ``RouteSnapshot`` dùng chung."""
        start_node = walk.nodes[0]
        step_coords: list[list[float]] = [
            [float(graph.coords[start_node, 0]), float(graph.coords[start_node, 1])]
        ]
        geom_coords: list[list[float]] = [
            [float(graph.coords[start_node, 0]), float(graph.coords[start_node, 1])]
        ]
        details: list[tuple[float, float | None, str]] = []
        route_ids: list[str] = [start]

        for edge in walk.edges:
            to_node = int(graph.dst[edge])
            end_lonlat = [float(graph.coords[to_node, 0]), float(graph.coords[to_node, 1])]
            step_coords.append(end_lonlat)
            via_raw = graph.via_nodes[edge] if hasattr(graph, "via_nodes") else ""
            if orig_coords is not None and via_raw:
                for token in via_raw.split(";"):
                    if not token:
                        continue
                    vid = int(token)
                    if 0 <= vid < len(orig_coords) and np.all(np.isfinite(orig_coords[vid])):
                        geom_coords.append(
                            [float(orig_coords[vid, 0]), float(orig_coords[vid, 1])]
                        )
            geom_coords.append(end_lonlat)
            w = float(graph.width[edge])
            details.append(
                (
                    float(graph.length[edge]),
                    w if math.isfinite(w) else None,
                    str(graph.road_label[edge]),
                )
            )
            for poi_idx in graph.pois_at.get(to_node, ()):
                pid = graph.poi_ids[poi_idx]
                if pid not in route_ids and pid != end:
                    route_ids.append(pid)

        route_ids.append(end)
        if len(step_coords) == 1:
            step_coords.append(step_coords[0][:])
            geom_coords.append(geom_coords[0][:])

        scale = np.array(
            [111320 * math.cos(math.radians(float(graph.coords[:, 1].mean()))), 111320.0]
        )
        collected = sorted(
            graph.poi_ids[i]
            for i in range(len(graph.poi_ids))
            if walk.collected_mask & (1 << i)
        )
        return {
            "route": route_ids,
            "score": float(walk.score),
            "distance_km": walk.distance_m / 1000.0,
            "duration_min": walk.duration_s / 60.0,
            "collected": collected,
            "geometry": {"type": "LineString", "coordinates": geom_coords},
            "narrow_m": float(walk.narrow_m),
            "unknown_width_m": float(walk.unknown_width_m),
            "instructions": road_instructions(step_coords, details, scale),
        }

    @staticmethod
    def _width_step_penalty(
        graph: GraphData, edges: np.ndarray, *, unknown_penalty: float = 0.0
    ) -> np.ndarray:
        """Tính phạt bước theo mức độ hẹp của mặt đường (< 5.5 m) và chiều dài cạnh."""
        w = graph.width[edges]
        finite_w = np.isfinite(w)
        narrow = finite_w & (w < NARROW_WIDTH_M)
        safe_w = np.where(narrow, w, NARROW_WIDTH_M)
        severity = 1.0 + (NARROW_WIDTH_M - safe_w) / NARROW_WIDTH_M
        length_km = graph.length[edges] / 1000.0
        return np.where(
            narrow,
            -3.0 * severity - 2.0 * severity * length_km,
            np.where(finite_w, 0.0, -unknown_penalty),
        )

    @classmethod
    def _edge_local_scores(
        cls,
        graph: GraphData,
        edges: np.ndarray,
        t: np.ndarray,
        to: np.ndarray,
        back: np.ndarray,
        cur: int,
        end_node: int,
        slot: str,
        here_poi: float,
        poi_to: np.ndarray,
        collected_mask: int,
        visited: set[int],
    ) -> np.ndarray:
        """Điểm hấp dẫn cục bộ cho ALNS/VNS với phạt đường hẹp theo độ hẹp thực tế."""
        n = len(edges)
        here_back = float(graph.to_target(slot, end_node)[cur])
        denom = np.maximum(t, 1e-6)
        target_prog = np.clip((here_back - back) / denom, -1.0, 1.0)
        both_finite = np.isfinite(here_poi) & np.isfinite(poi_to)
        safe_poi_to = np.where(both_finite, poi_to, 0.0)
        poi_prog = np.where(
            both_finite,
            np.clip(((here_poi if math.isfinite(here_poi) else 0.0) - safe_poi_to) / denom, -1.0, 1.0),
            0.0,
        )
        imm_poi = np.zeros(n, dtype=np.float64)
        for j in range(n):
            imm_poi[j] = (
                sum(
                    float(graph.poi_score[idx])
                    for idx in graph.pois_at.get(int(to[j]), ())
                    if not (collected_mask & (1 << idx))
                )
                / 5.0
            )
        revisit = np.array([1.0 if int(v) in visited else 0.0 for v in to], dtype=np.float64)
        width_pen = cls._width_step_penalty(graph, edges, unknown_penalty=0.0)
        return 10.0 * imm_poi + poi_prog + 0.01 * target_prog - 0.1 * revisit + width_pen

    @classmethod
    def _rollout_walk(
        cls,
        graph: GraphData,
        start_node: int,
        end_node: int,
        start_minute: int,
        budget_s: float,
        *,
        prefix_edges: Sequence[int] = (),
        rng: random.Random | None = None,
        mode: str = "greedy",
        score_fn: Callable[..., np.ndarray] | None = None,
        pheromone: dict[int, float] | None = None,
        alpha: float = 1.0,
        beta: float = 2.0,
        safety_margin: float = 1.05,
        max_steps: int = 1000,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> EdgeWalk | None:
        """Đi từng cạnh tới đích theo action mask an toàn và tra cứu POI điểm-tới-điểm."""
        slot0 = slot_of(start_minute)
        cur = start_node
        elapsed = 0.0
        distance_m = 0.0
        weighted_m = 0.0
        narrow_m = 0.0
        unknown_width_m = 0.0
        visited: set[int] = {start_node}
        collected_mask = cls._node_poi_mask(graph, start_node)
        walk_edges: list[int] = []
        walk_nodes: list[int] = [start_node]

        for raw_edge in prefix_edges:
            edge = int(raw_edge)
            if int(graph.src[edge]) != cur:
                return None
            slot = slot_of((start_minute + int(elapsed // 60)) % 1440)
            elapsed += float(graph.time[slot][edge])
            if elapsed > budget_s + 1e-6:
                return None
            length = float(graph.length[edge])
            width = float(graph.width[edge])
            is_narrow = bool(graph.narrow[edge])
            distance_m += length
            weighted_m += length * (2.0 if is_narrow else 1.0)
            if math.isfinite(width):
                if is_narrow:
                    narrow_m += length
            else:
                unknown_width_m += length
            cur = int(graph.dst[edge])
            visited.add(cur)
            collected_mask |= cls._node_poi_mask(graph, cur)
            walk_edges.append(edge)
            walk_nodes.append(cur)

        poi_table = graph.poi_distance_table(slot0)
        active_rows = graph.active_poi_rows(collected_mask, end_node)
        scorer = (
            score_fn
            if score_fn is not None
            else (
                StreetACOSolver._ppo_step_desirability
                if mode in ("aco", "aco_greedy")
                else cls._edge_local_scores
            )
        )
        steps = len(walk_edges)

        while cur != end_node and steps < max_steps:
            if cancelled():
                raise InterruptedError("Search cancelled")
            lo, hi = int(graph.indptr[cur]), int(graph.indptr[cur + 1])
            if lo == hi:
                return None
            edges = np.arange(lo, hi, dtype=np.int64)
            slot = slot_of((start_minute + int(elapsed // 60)) % 1440)
            t = graph.time[slot][edges]
            to = graph.dst[edges]
            back = graph.to_target(slot, end_node)[to]
            slack = budget_s - elapsed - t - back * safety_margin
            valid = slack >= 0
            if not valid.any():
                valid = np.zeros(len(edges), dtype=bool)
                valid[int(np.argmin(t + back))] = True

            valid_idx = np.flatnonzero(valid)
            if len(active_rows) == 0:
                here_poi = float("inf")
                poi_to = np.full(len(to), np.inf, dtype=np.float64)
            else:
                here_poi = float(poi_table[cur, active_rows].min())
                poi_to = poi_table[to][:, active_rows].min(axis=1)

            scores = scorer(
                graph,
                edges,
                t,
                to,
                back,
                cur,
                end_node,
                slot,
                here_poi,
                poi_to,
                collected_mask,
                visited,
            )

            if mode in ("greedy", "aco_greedy") or len(valid_idx) == 1:
                choice = int(valid_idx[int(np.argmax(scores[valid_idx]))])
            elif mode == "stochastic":
                assert rng is not None
                ranked = valid_idx[np.argsort(scores[valid_idx])[::-1]][
                    : min(3, len(valid_idx))
                ]
                top_scores = scores[ranked]
                weights = [
                    math.exp(max(-10.0, min(0.0, float(s - top_scores[0]))) / 0.35)
                    for s in top_scores
                ]
                choice = int(rng.choices(ranked.tolist(), weights=weights, k=1)[0])
            elif mode == "aco":
                assert rng is not None
                trail_map = pheromone if pheromone is not None else {}
                weights = []
                for idx in valid_idx:
                    edge_id = int(edges[idx])
                    tau = trail_map.get(edge_id, 1.0) ** alpha
                    eta = math.exp(max(-4.0, min(4.0, float(scores[idx])))) ** beta
                    weights.append(tau * eta)
                choice = int(rng.choices(valid_idx.tolist(), weights=weights, k=1)[0])
            else:
                raise ValueError(f"Unknown rollout mode: {mode}")

            edge = int(edges[choice])
            dt = float(t[choice])
            nxt = int(to[choice])
            elapsed += dt
            steps += 1
            if elapsed > budget_s + 1e-6:
                return None

            length = float(graph.length[edge])
            width = float(graph.width[edge])
            is_narrow = bool(graph.narrow[edge])
            distance_m += length
            weighted_m += length * (2.0 if is_narrow else 1.0)
            if math.isfinite(width):
                if is_narrow:
                    narrow_m += length
            else:
                unknown_width_m += length

            cur = nxt
            visited.add(cur)
            new_mask = collected_mask | cls._node_poi_mask(graph, cur)
            if new_mask != collected_mask:
                collected_mask = new_mask
                active_rows = graph.active_poi_rows(collected_mask, end_node)
            walk_edges.append(edge)
            walk_nodes.append(cur)

        if cur != end_node or elapsed > budget_s + 1e-6:
            return None
        return EdgeWalk(
            edges=tuple(walk_edges),
            nodes=tuple(walk_nodes),
            duration_s=elapsed,
            distance_m=distance_m,
            weighted_m=weighted_m,
            narrow_m=narrow_m,
            unknown_width_m=unknown_width_m,
            collected_mask=collected_mask,
            score=cls._mask_score(graph, collected_mask),
        )

    @classmethod
    def _prune_zero_poi_cycles(
        cls,
        graph: GraphData,
        walk: EdgeWalk,
        start_minute: int,
        budget_s: float,
        *,
        rank_fn: Callable[[EdgeWalk], Any] | None = None,
    ) -> tuple[EdgeWalk, Any]:
        """Cắt vòng lặp không thu thêm POI; lưu cache ``current_rank`` để tránh tính lại."""
        rank = rank_fn if rank_fn is not None else (lambda w: w.quality)
        current = walk
        current_rank = rank(current)
        for _ in range(16):
            first_seen: dict[int, int] = {}
            removed_span: tuple[int, int] | None = None
            for idx, node in enumerate(current.nodes):
                if node not in first_seen:
                    first_seen[node] = idx
                    continue
                prev_idx = first_seen[node]
                span_mask = 0
                for k in range(prev_idx, idx + 1):
                    span_mask |= cls._node_poi_mask(graph, current.nodes[k])
                outside_mask = cls._node_poi_mask(graph, current.nodes[0])
                for k in range(1, len(current.nodes)):
                    if not (prev_idx < k <= idx):
                        outside_mask |= cls._node_poi_mask(graph, current.nodes[k])
                if (outside_mask | span_mask) == outside_mask:
                    removed_span = (prev_idx, idx)
                    break
            if removed_span is None:
                break
            i, j = removed_span
            pruned_edges = current.edges[:i] + current.edges[j:]
            candidate = cls._evaluate_walk(
                graph,
                current.nodes[0],
                current.nodes[-1],
                pruned_edges,
                start_minute,
                budget_s,
            )
            if candidate is None:
                break
            cand_rank = rank(candidate)
            if cand_rank < current_rank:
                break
            current, current_rank = candidate, cand_rank
        return current, current_rank

    @classmethod
    def _initial_walk(
        cls,
        graph: GraphData,
        request: RouteRequest,
        *,
        max_steps: int,
    ) -> tuple[int, int, float, EdgeWalk]:
        direct = StreetDijkstraSolver(max_steps=max_steps).solve(
            StreetContext(graph=graph), request
        )
        start_node, _ = graph.point_by_id[request.start]
        end_node, _ = graph.point_by_id[request.end]
        budget_s = request.budget_s
        greedy = cls._rollout_walk(
            graph,
            start_node,
            end_node,
            request.start_minute,
            budget_s,
            mode="greedy",
            safety_margin=request.safety_margin,
            max_steps=max_steps,
            cancelled=request.cancelled,
        )
        best = direct
        if greedy is not None:
            greedy, _ = cls._prune_zero_poi_cycles(
                graph, greedy, request.start_minute, budget_s
            )
            if greedy.quality > best.quality:
                best = greedy
        return start_node, end_node, budget_s, best


class StreetDijkstraSolver(StreetBaseSolver):
    """Tìm lộ trình nhanh nhất theo thời gian phụ thuộc khung giờ trên ``GraphData``."""

    name: ClassVar[str] = "dijkstra"

    def solve(self, context: StreetContext, request: RouteRequest) -> EdgeWalk:
        graph = context.graph
        if request.start not in graph.point_by_id or request.end not in graph.point_by_id:
            raise ValueError("Start and end must be distinct available road points within 300 m")

        start_node, _ = graph.point_by_id[request.start]
        end_node, _ = graph.point_by_id[request.end]
        budget_s = request.budget_s

        best_time = np.full(graph.n, np.inf, dtype=np.float64)
        best_weight = np.full(graph.n, np.inf, dtype=np.float64)
        prev_node = np.full(graph.n, -1, dtype=np.int64)
        prev_edge = np.full(graph.n, -1, dtype=np.int64)

        best_time[start_node] = 0.0
        best_weight[start_node] = 0.0
        heap: list[tuple[float, float, int]] = [(0.0, 0.0, start_node)]

        while heap:
            request.check_cancelled()
            elapsed, w_m, u = heapq.heappop(heap)
            if elapsed > best_time[u] + 1e-9:
                continue
            if elapsed == best_time[u] and w_m > best_weight[u] + 1e-9:
                continue
            if u == end_node:
                break
            slot = slot_of((request.start_minute + int(elapsed // 60)) % 1440)
            lo, hi = int(graph.indptr[u]), int(graph.indptr[u + 1])
            times = graph.time[slot]
            for edge in range(lo, hi):
                new_t = elapsed + float(times[edge])
                if new_t > budget_s + 1e-6:
                    continue
                v = int(graph.dst[edge])
                new_w = w_m + float(graph.length[edge]) * (2.0 if graph.narrow[edge] else 1.0)
                if new_t < best_time[v] - 1e-9 or (
                    abs(new_t - best_time[v]) <= 1e-9 and new_w < best_weight[v] - 1e-9
                ):
                    best_time[v] = new_t
                    best_weight[v] = new_w
                    prev_node[v] = u
                    prev_edge[v] = edge
                    heapq.heappush(heap, (new_t, new_w, v))

        if not math.isfinite(best_time[end_node]) or best_time[end_node] > budget_s + 1e-6:
            raise NoRoadRouteError("No directed route found within the time budget")

        edges: list[int] = []
        cur = end_node
        while cur != start_node:
            edges.append(int(prev_edge[cur]))
            cur = int(prev_node[cur])
        edges.reverse()
        walk = self._evaluate_walk(
            graph, start_node, end_node, edges, request.start_minute, budget_s
        )
        if walk is None:
            raise NoRoadRouteError("No directed route found within the time budget")
        return walk


class StreetALNSSolver(StreetBaseSolver):
    """Tìm lộ trình bằng ALNS ở mức cạnh (cắt đoạn/đuôi + nối lại theo action mask)."""

    name: ClassVar[str] = "alns"

    @classmethod
    def _choose_cut_index(
        cls, graph: GraphData, walk: EdgeWalk, rng: random.Random, worst: bool
    ) -> int:
        m = len(walk.edges)
        if m <= 2:
            return 0
        if not worst:
            lo = max(0, int(0.35 * m))
            hi = max(lo + 1, int(0.85 * m))
            return rng.randint(lo, min(hi, m - 1))

        seen_mask = cls._node_poi_mask(graph, walk.nodes[0])
        cum_scores = [0.0]
        for node in walk.nodes[1:]:
            node_mask = cls._node_poi_mask(graph, node)
            new_bits = node_mask & ~seen_mask
            seen_mask |= node_mask
            cum_scores.append(cum_scores[-1] + cls._mask_score(graph, new_bits))
        lo = max(1, int(0.25 * m))
        hi = max(lo + 1, int(0.85 * m))
        best_cut = lo
        lowest_density = float("inf")
        total_score = cum_scores[-1]
        for cut in range(lo, min(hi, m)):
            suffix_gain = total_score - cum_scores[cut]
            suffix_steps = max(1, m - cut)
            density = suffix_gain / suffix_steps
            if density < lowest_density:
                lowest_density = density
                best_cut = cut
        return best_cut

    @classmethod
    def _fork_worker(cls, task: tuple[Any, ...]) -> tuple[int, int, EdgeWalk | None]:
        (
            trial_seed,
            start_node,
            end_node,
            start_minute,
            budget_s,
            current,
            destroy_weights,
            repair_weights,
            safety_margin,
            max_steps,
        ) = task
        graph = cls._fork_state["graph"]
        rng = random.Random(trial_seed)
        destroy = rng.choices(range(2), weights=destroy_weights)[0]
        repair = rng.choices(range(2), weights=repair_weights)[0]
        cut = cls._choose_cut_index(graph, current, rng, worst=bool(destroy))
        candidate = cls._rollout_walk(
            graph,
            start_node,
            end_node,
            start_minute,
            budget_s,
            prefix_edges=current.edges[:cut],
            rng=rng,
            mode="stochastic" if repair else "greedy",
            safety_margin=safety_margin,
            max_steps=max_steps,
        )
        if candidate is not None:
            candidate, _ = cls._prune_zero_poi_cycles(
                graph, candidate, start_minute, budget_s
            )
        return destroy, repair, candidate

    def solve(self, context: StreetContext, request: RouteRequest) -> EdgeWalk:
        graph = context.graph
        start_node, end_node, budget_s, best = self._initial_walk(
            graph, request, max_steps=self.max_steps
        )
        current = best
        rng = random.Random(request.seed)
        destroy_weights, repair_weights = [1.0, 1.0], [1.0, 1.0]

        if self._can_fork(request.workers, request.iterations):
            self._prewarm_graph(graph, end_node, request.start_minute)
            ctx = mp.get_context("fork")
            self._fork_state["graph"] = graph
            try:
                with ctx.Pool(processes=min(request.workers, request.iterations)) as pool:
                    done = 0
                    while done < request.iterations:
                        request.check_cancelled()
                        batch = min(request.workers, request.iterations - done)
                        tasks = [
                            (
                                type(self),
                                (
                                    request.seed * 1000003 + done + offset,
                                    start_node,
                                    end_node,
                                    request.start_minute,
                                    budget_s,
                                    current,
                                    list(destroy_weights),
                                    list(repair_weights),
                                    request.safety_margin,
                                    self.max_steps,
                                ),
                            )
                            for offset in range(batch)
                        ]
                        best_cand: EdgeWalk | None = None
                        for destroy, repair, candidate in pool.map(
                            _fork_worker_dispatch, tasks
                        ):
                            improved_best = False
                            improved_current = False
                            if candidate is not None:
                                improved_best = candidate.quality > best.quality
                                improved_current = candidate.quality > current.quality
                                if improved_best:
                                    best = candidate
                                if best_cand is None or candidate.quality > best_cand.quality:
                                    best_cand = candidate
                            reward = 5.0 if improved_best else 2.0 if improved_current else 0.5
                            destroy_weights[destroy] = (
                                0.8 * destroy_weights[destroy] + 0.2 * reward
                            )
                            repair_weights[repair] = (
                                0.8 * repair_weights[repair] + 0.2 * reward
                            )
                        if best_cand is not None:
                            temperature = max(
                                0.01, 1.0 - done / max(request.iterations, 1)
                            )
                            delta = best_cand.score - current.score
                            if best_cand.quality > current.quality or (
                                delta < 0 and rng.random() < math.exp(delta / temperature)
                            ):
                                current = best_cand
                        done += batch
            finally:
                self._fork_state.clear()
            return best

        for iteration in range(request.iterations):
            destroy = rng.choices(range(2), weights=destroy_weights)[0]
            repair = rng.choices(range(2), weights=repair_weights)[0]
            cut = self._choose_cut_index(graph, current, rng, worst=bool(destroy))
            candidate = self._rollout_walk(
                graph,
                start_node,
                end_node,
                request.start_minute,
                budget_s,
                prefix_edges=current.edges[:cut],
                rng=rng,
                mode="stochastic" if repair else "greedy",
                safety_margin=request.safety_margin,
                max_steps=self.max_steps,
                cancelled=request.cancelled,
            )
            improved_best = False
            improved_current = False
            if candidate is not None:
                candidate, _ = self._prune_zero_poi_cycles(
                    graph, candidate, request.start_minute, budget_s
                )
                improved_best = candidate.quality > best.quality
                improved_current = candidate.quality > current.quality
                temperature = max(0.01, 1.0 - iteration / max(request.iterations, 1))
                delta = candidate.score - current.score
                if improved_current or (
                    delta < 0 and rng.random() < math.exp(delta / temperature)
                ):
                    current = candidate
                if improved_best:
                    best = candidate
            reward = 5.0 if improved_best else 2.0 if improved_current else 0.5
            destroy_weights[destroy] = 0.8 * destroy_weights[destroy] + 0.2 * reward
            repair_weights[repair] = 0.8 * repair_weights[repair] + 0.2 * reward
        return best


class StreetVNSSolver(StreetBaseSolver):
    """Tìm lộ trình bằng VNS ở mức cạnh (lắc theo 4 mức độ dài + tỉa chu trình + tối ưu đuôi)."""

    name: ClassVar[str] = "vns"

    @classmethod
    def _fork_worker(cls, task: tuple[Any, ...]) -> EdgeWalk | None:
        (
            trial_seed,
            start_node,
            end_node,
            start_minute,
            budget_s,
            best,
            neighbourhood,
            safety_margin,
            max_steps,
        ) = task
        graph = cls._fork_state["graph"]
        rng = random.Random(trial_seed)
        m = len(best.edges)
        drop_frac = min(0.70, 0.15 * neighbourhood)
        cut = max(0, int(m * (1.0 - drop_frac)))
        candidate = cls._rollout_walk(
            graph,
            start_node,
            end_node,
            start_minute,
            budget_s,
            prefix_edges=best.edges[:cut],
            rng=rng,
            mode="stochastic",
            safety_margin=safety_margin,
            max_steps=max_steps,
        )
        if candidate is None:
            return None
        local_best, _ = cls._prune_zero_poi_cycles(
            graph, candidate, start_minute, budget_s
        )
        lm = len(local_best.edges)
        if lm >= 4:
            for frac in (0.25, 0.50, 0.75):
                pivot = int(lm * frac)
                refined = cls._rollout_walk(
                    graph,
                    start_node,
                    end_node,
                    start_minute,
                    budget_s,
                    prefix_edges=local_best.edges[:pivot],
                    mode="greedy",
                    safety_margin=safety_margin,
                    max_steps=max_steps,
                )
                if refined is not None:
                    refined, _ = cls._prune_zero_poi_cycles(
                        graph, refined, start_minute, budget_s
                    )
                    if refined.quality > local_best.quality:
                        local_best = refined
        return local_best

    def solve(self, context: StreetContext, request: RouteRequest) -> EdgeWalk:
        graph = context.graph
        start_node, end_node, budget_s, best = self._initial_walk(
            graph, request, max_steps=self.max_steps
        )
        rng = random.Random(request.seed)
        neighbourhood = 1

        if self._can_fork(request.workers, request.iterations):
            self._prewarm_graph(graph, end_node, request.start_minute)
            ctx = mp.get_context("fork")
            self._fork_state["graph"] = graph
            try:
                with ctx.Pool(processes=min(request.workers, request.iterations)) as pool:
                    done = 0
                    while done < request.iterations:
                        request.check_cancelled()
                        batch = min(request.workers, request.iterations - done)
                        tasks = [
                            (
                                type(self),
                                (
                                    request.seed * 1000003 + done + offset,
                                    start_node,
                                    end_node,
                                    request.start_minute,
                                    budget_s,
                                    best,
                                    ((neighbourhood - 1 + offset) % 4) + 1,
                                    request.safety_margin,
                                    self.max_steps,
                                ),
                            )
                            for offset in range(batch)
                        ]
                        candidates = [
                            c
                            for c in pool.map(_fork_worker_dispatch, tasks)
                            if c is not None
                        ]
                        winner = (
                            max(candidates, key=lambda w: w.quality)
                            if candidates
                            else None
                        )
                        if winner is not None and winner.quality > best.quality:
                            best = winner
                            neighbourhood = 1
                        else:
                            neighbourhood = ((neighbourhood - 1 + batch) % 4) + 1
                        done += batch
            finally:
                self._fork_state.clear()
            return best

        for _ in range(request.iterations):
            m = len(best.edges)
            drop_frac = min(0.70, 0.15 * neighbourhood)
            cut = max(0, int(m * (1.0 - drop_frac)))
            candidate = self._rollout_walk(
                graph,
                start_node,
                end_node,
                request.start_minute,
                budget_s,
                prefix_edges=best.edges[:cut],
                rng=rng,
                mode="stochastic",
                safety_margin=request.safety_margin,
                max_steps=self.max_steps,
                cancelled=request.cancelled,
            )
            if candidate is None:
                neighbourhood = neighbourhood % 4 + 1
                continue

            local_best, _ = self._prune_zero_poi_cycles(
                graph, candidate, request.start_minute, budget_s
            )
            lm = len(local_best.edges)
            if lm >= 4:
                for frac in (0.25, 0.50, 0.75):
                    pivot = int(lm * frac)
                    refined = self._rollout_walk(
                        graph,
                        start_node,
                        end_node,
                        request.start_minute,
                        budget_s,
                        prefix_edges=local_best.edges[:pivot],
                        mode="greedy",
                        safety_margin=request.safety_margin,
                        max_steps=self.max_steps,
                        cancelled=request.cancelled,
                    )
                    if refined is not None:
                        refined, _ = self._prune_zero_poi_cycles(
                            graph, refined, request.start_minute, budget_s
                        )
                        if refined.quality > local_best.quality:
                            local_best = refined

            if local_best.quality > best.quality:
                best = local_best
                neighbourhood = 1
            else:
                neighbourhood = neighbourhood % 4 + 1
        return best


class StreetACOSolver(StreetBaseSolver):
    """Ant Colony Optimization ở mức cạnh tối ưu tổng phần thưởng ``BusRouteEnv`` của PPO."""

    name: ClassVar[str] = "aco"

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation: float = 0.9,
        tau_min: float = 0.1,
        deposit: float = 1.0,
        max_steps: int = 1000,
    ) -> None:
        super().__init__(max_steps=max_steps)
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.tau_min = tau_min
        self.deposit = deposit

    @classmethod
    def _ppo_step_desirability(
        cls,
        graph: GraphData,
        edges: np.ndarray,
        t: np.ndarray,
        to: np.ndarray,
        back: np.ndarray,
        cur: int,
        end_node: int,
        slot: str,
        here_poi: float,
        poi_to: np.ndarray,
        collected_mask: int,
        visited: set[int],
    ) -> np.ndarray:
        """Tính độ hấp dẫn cục bộ bất biến theo tỷ lệ thời gian cho kiến ACO.

        Giữ nguyên ưu tiên của ``BusRouteEnv`` (POI mới, tiến về POI gần nhất,
        tránh quay vòng và né đường hẹp/chưa rõ bề rộng) ở thang phù hợp cho
        ``exp(score) ** beta`` thay vì phần thưởng 1-bước thô dễ bị triệt tiêu trên
        cạnh ngắn.
        """
        n = len(edges)
        here_back = float(graph.to_target(slot, end_node)[cur])
        denom = np.maximum(t, 1e-6)
        target_prog = np.clip((here_back - back) / denom, -1.0, 1.0)
        both_finite = np.isfinite(here_poi) & np.isfinite(poi_to)
        safe_poi_to = np.where(both_finite, poi_to, 0.0)
        poi_prog = np.where(
            both_finite,
            np.clip(((here_poi if math.isfinite(here_poi) else 0.0) - safe_poi_to) / denom, -1.0, 1.0),
            0.0,
        )
        imm_poi = np.zeros(n, dtype=np.float64)
        for j in range(n):
            imm_poi[j] = (
                sum(
                    float(graph.poi_score[idx])
                    for idx in graph.pois_at.get(int(to[j]), ())
                    if not (collected_mask & (1 << idx))
                )
                / 5.0
            )
        revisit = np.array([1.0 if int(v) in visited else 0.0 for v in to], dtype=np.float64)
        target_weight = 0.02 if math.isfinite(here_poi) else 1.0
        width_pen = cls._width_step_penalty(graph, edges, unknown_penalty=0.05)
        return (
            10.0 * imm_poi
            + 1.2 * poi_prog
            + target_weight * target_prog
            - 0.5 * revisit
            + width_pen
        )

    @classmethod
    def _walk_ppo_return(
        cls, graph: GraphData, walk: EdgeWalk, start_minute: int
    ) -> float:
        """Tính chính xác tổng phần thưởng episode của ``BusRouteEnv`` cho một ``EdgeWalk``."""
        slot0 = slot_of(start_minute)
        walk_dists = graph.poi_distance_table(slot0)[np.asarray(walk.nodes, dtype=np.int64)]
        cur = walk.nodes[0]
        end_node = walk.nodes[-1]
        elapsed = 0.0
        visited: set[int] = {cur}
        collected_mask = cls._node_poi_mask(graph, cur)
        active_rows = graph.active_poi_rows(collected_mask, end_node)
        here_poi = float(walk_dists[0, active_rows].min()) if len(active_rows) else float("inf")
        total = 0.0
        for step, edge in enumerate(walk.edges):
            slot = slot_of((start_minute + int(elapsed // 60)) % 1440)
            dt = float(graph.time[slot][edge])
            nxt = int(graph.dst[edge])
            nxt_poi = (
                float(walk_dists[step + 1, active_rows].min())
                if len(active_rows)
                else float("inf")
            )
            shaping = 0.0
            if math.isfinite(here_poi) and math.isfinite(nxt_poi):
                shaping = PPO_PROGRESS_COEF * (here_poi - nxt_poi) / 60.0

            length_km = float(graph.length[edge]) / 1000.0
            w = float(graph.width[edge])
            if math.isfinite(w):
                width_reward = (
                    -PPO_NARROW_PENALTY
                    * length_km
                    * (1.0 + (NARROW_WIDTH_M - w) / NARROW_WIDTH_M)
                    if w < NARROW_WIDTH_M
                    else 0.0
                )
            else:
                width_reward = -PPO_UNKNOWN_WIDTH_PENALTY * length_km

            reward = (
                -PPO_TIME_PENALTY * (dt / 60.0)
                + shaping
                + width_reward
                - (PPO_REVISIT_PENALTY if nxt in visited else 0.0)
            )
            visited.add(nxt)
            elapsed += dt
            cur = nxt
            new_mask = collected_mask | cls._node_poi_mask(graph, cur)
            if new_mask != collected_mask:
                reward += cls._mask_score(graph, new_mask & ~collected_mask)
                collected_mask = new_mask
                active_rows = graph.active_poi_rows(collected_mask, end_node)
                here_poi = (
                    float(walk_dists[step + 1, active_rows].min())
                    if len(active_rows)
                    else float("inf")
                )
            else:
                here_poi = nxt_poi
            if cur == end_node:
                reward += PPO_SUCCESS_BONUS
            total += reward
        return total

    @classmethod
    def _walk_key(
        cls, graph: GraphData, walk: EdgeWalk, start_minute: int
    ) -> tuple[float, tuple[float, float, float, float]]:
        """Khóa xếp hạng của ACO theo tổng phần thưởng PPO, dùng quality để phá thế hòa."""
        return (round(cls._walk_ppo_return(graph, walk, start_minute), 9), walk.quality)

    @classmethod
    def _initial_aco_walk(
        cls,
        graph: GraphData,
        request: RouteRequest,
        *,
        max_steps: int,
    ) -> tuple[int, int, float, EdgeWalk, tuple[float, tuple[float, float, float, float]]]:
        """Khởi tạo hạt giống cho ACO và trả về kèm khóa ``_walk_key`` để không tính lại."""
        direct = StreetDijkstraSolver(max_steps=max_steps).solve(
            StreetContext(graph=graph), request
        )
        start_node, _ = graph.point_by_id[request.start]
        end_node, _ = graph.point_by_id[request.end]
        budget_s = request.budget_s
        greedy = cls._rollout_walk(
            graph,
            start_node,
            end_node,
            request.start_minute,
            budget_s,
            mode="aco_greedy",
            score_fn=cls._ppo_step_desirability,
            safety_margin=request.safety_margin,
            max_steps=max_steps,
            cancelled=request.cancelled,
        )
        rank = lambda w: cls._walk_key(graph, w, request.start_minute)
        best = direct
        best_key = rank(direct)
        if greedy is not None:
            greedy, greedy_key = cls._prune_zero_poi_cycles(
                graph, greedy, request.start_minute, budget_s, rank_fn=rank
            )
            if greedy_key > best_key:
                best, best_key = greedy, greedy_key
        return start_node, end_node, budget_s, best, best_key

    @classmethod
    def _fork_worker(
        cls, task: tuple[Any, ...]
    ) -> tuple[EdgeWalk, tuple[float, tuple[float, float, float, float]]] | None:
        (
            ant_seed,
            start_node,
            end_node,
            start_minute,
            budget_s,
            pheromone,
            alpha,
            beta,
            safety_margin,
            max_steps,
        ) = task
        graph = cls._fork_state["graph"]
        rng = random.Random(ant_seed)
        candidate = cls._rollout_walk(
            graph,
            start_node,
            end_node,
            start_minute,
            budget_s,
            rng=rng,
            mode="aco",
            score_fn=cls._ppo_step_desirability,
            pheromone=pheromone,
            alpha=alpha,
            beta=beta,
            safety_margin=safety_margin,
            max_steps=max_steps,
        )
        if candidate is None:
            return None
        return cls._prune_zero_poi_cycles(
            graph,
            candidate,
            start_minute,
            budget_s,
            rank_fn=lambda w: cls._walk_key(graph, w, start_minute),
        )

    def solve(self, context: StreetContext, request: RouteRequest) -> EdgeWalk:
        graph = context.graph
        start_node, end_node, budget_s, best, best_key = self._initial_aco_walk(
            graph, request, max_steps=self.max_steps
        )
        rank = lambda w: self._walk_key(graph, w, request.start_minute)
        rng = random.Random(request.seed)
        pheromone: dict[int, float] = {}
        for edge in best.edges:
            pheromone[edge] = 1.0 + self.deposit

        if self._can_fork(request.workers, request.iterations):
            self._prewarm_graph(graph, end_node, request.start_minute)
            ctx = mp.get_context("fork")
            self._fork_state["graph"] = graph
            try:
                with ctx.Pool(processes=min(request.workers, request.iterations)) as pool:
                    done = 0
                    while done < request.iterations:
                        request.check_cancelled()
                        batch = min(request.workers, request.iterations - done)
                        snap_pheromone = dict(pheromone)
                        tasks = [
                            (
                                type(self),
                                (
                                    request.seed * 1000003 + done + offset,
                                    start_node,
                                    end_node,
                                    request.start_minute,
                                    budget_s,
                                    snap_pheromone,
                                    self.alpha,
                                    self.beta,
                                    request.safety_margin,
                                    self.max_steps,
                                ),
                            )
                            for offset in range(batch)
                        ]
                        for outcome in pool.map(_fork_worker_dispatch, tasks):
                            if outcome is not None:
                                candidate, cand_key = outcome
                                if cand_key > best_key:
                                    best, best_key = candidate, cand_key
                        for edge, trail in list(pheromone.items()):
                            pheromone[edge] = max(
                                self.tau_min, trail * (self.evaporation**batch)
                            )
                        for edge in best.edges:
                            pheromone[edge] = (
                                pheromone.get(edge, 1.0) + self.deposit * batch
                            )
                        done += batch
            finally:
                self._fork_state.clear()
            return best

        for _ in range(request.iterations):
            candidate = self._rollout_walk(
                graph,
                start_node,
                end_node,
                request.start_minute,
                budget_s,
                rng=rng,
                mode="aco",
                score_fn=self._ppo_step_desirability,
                pheromone=pheromone,
                alpha=self.alpha,
                beta=self.beta,
                safety_margin=request.safety_margin,
                max_steps=self.max_steps,
                cancelled=request.cancelled,
            )
            if candidate is not None:
                candidate, cand_key = self._prune_zero_poi_cycles(
                    graph, candidate, request.start_minute, budget_s, rank_fn=rank
                )
                if cand_key > best_key:
                    best, best_key = candidate, cand_key
            for edge, trail in list(pheromone.items()):
                pheromone[edge] = max(self.tau_min, trail * self.evaporation)
            for edge in best.edges:
                pheromone[edge] = pheromone.get(edge, 1.0) + self.deposit
        return best


STREET_SOLVERS: dict[str, type[StreetBaseSolver]] = {
    "dijkstra": StreetDijkstraSolver,
    "alns": StreetALNSSolver,
    "vns": StreetVNSSolver,
    "aco": StreetACOSolver,
}


# Các hàm bọc tương thích ngược cho caller hiện có và bộ kiểm thử.


def evaluate_edge_walk(
    graph: GraphData,
    start_node: int,
    end_node: int,
    edges: Sequence[int],
    start_minute: int,
    budget_s: float,
) -> EdgeWalk | None:
    """Kiểm tra và tính toàn bộ chỉ số cho một chuỗi cạnh trên ``GraphData``."""
    return StreetBaseSolver._evaluate_walk(
        graph, start_node, end_node, edges, start_minute, budget_s
    )


def walk_to_snapshot(
    graph: GraphData,
    walk: EdgeWalk,
    start: str,
    end: str,
    *,
    orig_coords: np.ndarray | None = None,
) -> dict[str, Any]:
    """Chuyển ``EdgeWalk`` sang định dạng ``RouteSnapshot`` dùng chung."""
    return StreetBaseSolver._walk_to_snapshot(
        graph, walk, start, end, orig_coords=orig_coords
    )


def _walk_ppo_return(graph: GraphData, walk: EdgeWalk, start_minute: int) -> float:
    return StreetACOSolver._walk_ppo_return(graph, walk, start_minute)


def _prune_zero_poi_cycles(
    graph: GraphData,
    walk: EdgeWalk,
    start_minute: int,
    budget_s: float,
    *,
    rank_fn: Callable[[EdgeWalk], Any] | None = None,
) -> EdgeWalk:
    pruned, _ = StreetBaseSolver._prune_zero_poi_cycles(
        graph, walk, start_minute, budget_s, rank_fn=rank_fn
    )
    return pruned


def edge_dijkstra(
    graph: GraphData,
    start: str,
    end: str,
    budget_min: float,
    start_minute: int = 480,
    *,
    cancelled: Callable[[], bool] = lambda: False,
) -> EdgeWalk:
    """Tìm lộ trình nhanh nhất theo thời gian phụ thuộc khung giờ trực tiếp trên ``GraphData``."""
    request = RouteRequest(
        start=start,
        end=end,
        budget_min=budget_min,
        start_minute=start_minute,
        cancelled=cancelled,
    )
    return StreetDijkstraSolver().solve(StreetContext(graph=graph), request)


def edge_aco(
    graph: GraphData,
    start: str,
    end: str,
    budget_min: float,
    start_minute: int = 480,
    *,
    iterations: int = 100,
    seed: int = 0,
    workers: int = 1,
    safety_margin: float = 1.05,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation: float = 0.9,
    tau_min: float = 0.1,
    deposit: float = 1.0,
    max_steps: int = 1000,
    cancelled: Callable[[], bool] = lambda: False,
) -> EdgeWalk:
    """Tìm lộ trình bằng Ant Colony Optimization từng cạnh với bảng pheromone và phần thưởng PPO."""
    request = RouteRequest(
        start=start,
        end=end,
        budget_min=budget_min,
        start_minute=start_minute,
        iterations=iterations,
        seed=seed,
        workers=workers,
        safety_margin=safety_margin,
        cancelled=cancelled,
    )
    solver = StreetACOSolver(
        alpha=alpha,
        beta=beta,
        evaporation=evaporation,
        tau_min=tau_min,
        deposit=deposit,
        max_steps=max_steps,
    )
    return solver.solve(StreetContext(graph=graph), request)


def edge_alns(
    graph: GraphData,
    start: str,
    end: str,
    budget_min: float,
    start_minute: int = 480,
    *,
    iterations: int = 100,
    seed: int = 0,
    workers: int = 1,
    safety_margin: float = 1.05,
    max_steps: int = 1000,
    cancelled: Callable[[], bool] = lambda: False,
) -> EdgeWalk:
    """Tìm lộ trình bằng ALNS ở mức cạnh (cắt đoạn/đuôi + nối lại theo action mask)."""
    request = RouteRequest(
        start=start,
        end=end,
        budget_min=budget_min,
        start_minute=start_minute,
        iterations=iterations,
        seed=seed,
        workers=workers,
        safety_margin=safety_margin,
        cancelled=cancelled,
    )
    return StreetALNSSolver(max_steps=max_steps).solve(
        StreetContext(graph=graph), request
    )


def edge_vns(
    graph: GraphData,
    start: str,
    end: str,
    budget_min: float,
    start_minute: int = 480,
    *,
    iterations: int = 100,
    seed: int = 0,
    workers: int = 1,
    safety_margin: float = 1.05,
    max_steps: int = 1000,
    cancelled: Callable[[], bool] = lambda: False,
) -> EdgeWalk:
    """Tìm lộ trình bằng VNS ở mức cạnh (lắc theo 4 mức độ dài + tỉa chu trình + tối ưu đuôi)."""
    request = RouteRequest(
        start=start,
        end=end,
        budget_min=budget_min,
        start_minute=start_minute,
        iterations=iterations,
        seed=seed,
        workers=workers,
        safety_margin=safety_margin,
        cancelled=cancelled,
    )
    return StreetVNSSolver(max_steps=max_steps).solve(
        StreetContext(graph=graph), request
    )


def find_edge_route(
    nodes_path: str | Path,
    edges_path: str | Path,
    widths_path: str | Path,
    points: Sequence[dict[str, Any]],
    start: str,
    end: str,
    cancelled: Callable[[], bool] = lambda: False,
    *,
    algorithm: str = "dijkstra",
    budget_min: float = 240.0,
    departure: str = "08:00",
    iterations: int = 100,
    seed: int = 0,
    workers: int = 1,
    safety_margin: float = 1.05,
    reduced_dir: Path = REDUCED_DIR,
    use_reduced: bool = True,
    graph: GraphData | None = None,
    orig_coords: np.ndarray | None = None,
) -> dict[str, Any]:
    """Giải bài toán định tuyến ở mức cạnh cho ``dijkstra``, ``alns``, ``vns`` hoặc ``aco``."""
    solver_cls = STREET_SOLVERS.get(algorithm)
    if solver_cls is None:
        raise ValueError(f"Unsupported edge algorithm: {algorithm}")
    request = RouteRequest.from_departure(
        start,
        end,
        budget_min,
        departure=departure,
        iterations=iterations,
        seed=seed,
        workers=workers,
        safety_margin=safety_margin,
        cancelled=cancelled,
    )
    solver = solver_cls()
    context = (
        StreetContext(graph=graph, orig_coords=orig_coords)
        if graph is not None
        else None
    )
    data = (
        None
        if context is not None
        else RoadDatasetSpec.from_raw(
            nodes_path,
            edges_path,
            widths_path,
            points,
            reduced_dir=reduced_dir,
            use_reduced=use_reduced,
        )
    )
    return solver.solve_snapshot(request, data=data, context=context)
