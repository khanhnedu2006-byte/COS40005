"""PPO ở mức waypoint (chọn POI kế tiếp) trên ``RoadGraph`` với action masking ngân sách.

Môi trường ``WaypointBusEnv`` đồng bộ không gian hành động với ALNS/VNS/ACO: mỗi bước
chọn 1 POI tiếp theo (hoặc kết thúc về đích B) thay vì chọn từng cạnh đường.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from stable_baselines3.common.vec_env import DummyVecEnv

from busroute.graph.ab_estimator import SLOTS
from busroute.routing.base import RoadDatasetSpec, RouteRequest
from busroute.routing.ppo_env import REDUCED_DIR
from busroute.routing.ppo_train import compute_gae, run_episode, update_policy
from busroute.routing.road_graph import (
    RoadGraph,
    RoadProblem,
    RoadResult,
    _parallel_edges,
    _road_leg,
)
from busroute.routing.road_metaheuristics import PoiBaseSolver, PoiContext
from busroute.utils.timeslots import slot_of

CHECKPOINT_WAYPOINT = (
    Path(__file__).resolve().parents[3] / "checkpoints" / "ppo_waypoint_reduced.zip"
)
WAYPOINT_CAND_FEATURES = 8
WAYPOINT_GLOBAL_FEATURES = 8


class WaypointTable:
    """Bảng chặng mọi cặp điểm đã snap trên ``RoadGraph`` phục vụ train/eval nhanh."""

    def __init__(
        self,
        point_ids: list[str],
        poi_ids: list[str],
        poi_bits: list[int],
        poi_scores: np.ndarray,
        snapped_nodes: dict[str, int],
        node_poi_masks: dict[str, int],
        dur: dict[str, np.ndarray],
        dist: dict[str, np.ndarray],
        narrow: dict[str, np.ndarray],
        masks: dict[str, list[list[int]]],
    ) -> None:
        self.point_ids = point_ids
        self.point_index = {pid: idx for idx, pid in enumerate(point_ids)}
        self.poi_ids = poi_ids
        self.poi_bits = poi_bits
        self.poi_scores = poi_scores
        self.all_poi_scores = np.array(
            [float(s) for s in poi_scores], dtype=np.float64
        )
        self.snapped_nodes = snapped_nodes
        self.node_poi_masks = node_poi_masks
        self.dur = dur
        self.dist = dist
        self.narrow = narrow
        self.masks = masks
        # Ánh xạ bit POI trên RoadGraph sang giá trị điểm.
        self.bit_score: dict[int, float] = {
            bit: float(score) for bit, score in zip(poi_bits, poi_scores)
        }

    def mask_score(self, mask: int) -> float:
        """Tính tổng điểm của các bit POI đang bật trong ``mask``."""
        return sum(
            score for bit, score in self.bit_score.items() if mask & (1 << bit)
        )

    @classmethod
    def from_road_graph(cls, road_graph: RoadGraph, *, workers: int = 1) -> WaypointTable:
        """Tiền tính thời gian, độ dài, đường hẹp và bitmask POI cho toàn bộ 5 khung giờ."""
        point_ids = sorted(road_graph.snapped.keys())
        point_idx = {pid: idx for idx, pid in enumerate(point_ids)}
        poi_pairs = [
            (bit, pid)
            for bit, pid in enumerate(road_graph.pois)
            if pid in road_graph.snapped
        ]
        poi_ids = [pid for _, pid in poi_pairs]
        poi_bits = [bit for bit, _ in poi_pairs]
        poi_scores = np.array(
            [float(road_graph.scores[bit]) for bit, _ in poi_pairs], dtype=np.float64
        )
        snapped_nodes = {pid: int(road_graph.snapped[pid]) for pid in point_ids}
        node_poi_masks = {
            pid: int(road_graph.node_mask(snapped_nodes[pid])) for pid in point_ids
        }

        n_pts = len(point_ids)
        unique_nodes = sorted(set(snapped_nodes.values()))
        dur: dict[str, np.ndarray] = {}
        dist: dict[str, np.ndarray] = {}
        narrow: dict[str, np.ndarray] = {}
        masks: dict[str, list[list[int]]] = {}

        for slot in SLOTS:
            selected = _parallel_edges(road_graph, slot)
            edge_of = {
                (int(road_graph.u[e]), int(road_graph.v[e])): int(e) for e in selected
            }
            matrix = csr_matrix(
                (
                    road_graph.times[slot][selected],
                    (road_graph.u[selected], road_graph.v[selected]),
                ),
                shape=(len(road_graph.nodes), len(road_graph.nodes)),
            )
            node_trees = {}
            if workers > 1 and len(unique_nodes) > 1:
                with ThreadPoolExecutor(max_workers=min(workers, len(unique_nodes))) as pool:
                    tree_results = list(
                        pool.map(
                            lambda src_node: dijkstra(
                                matrix,
                                directed=True,
                                indices=src_node,
                                return_predecessors=True,
                            ),
                            unique_nodes,
                        )
                    )
                node_trees = dict(zip(unique_nodes, tree_results))
            else:
                for src_node in unique_nodes:
                    node_trees[src_node] = dijkstra(
                        matrix, directed=True, indices=src_node, return_predecessors=True
                    )

            d_mat = np.full((n_pts, n_pts), np.inf, dtype=np.float64)
            dist_mat = np.zeros((n_pts, n_pts), dtype=np.float64)
            nar_mat = np.zeros((n_pts, n_pts), dtype=np.float64)
            mask_mat: list[list[int]] = [[0] * n_pts for _ in range(n_pts)]

            for a in point_ids:
                ia = point_idx[a]
                src_node = snapped_nodes[a]
                dists_arr, preds_arr = node_trees[src_node]
                src_mask = node_poi_masks[a]
                for b in point_ids:
                    ib = point_idx[b]
                    dst_node = snapped_nodes[b]
                    leg = _road_leg(
                        road_graph,
                        src_node,
                        dst_node,
                        dists_arr,
                        preds_arr,
                        edge_of,
                        src_mask,
                    )
                    if leg is None:
                        continue
                    d_mat[ia, ib] = leg.duration_s
                    dist_mat[ia, ib] = leg.distance_m
                    nar_mat[ia, ib] = leg.weighted_m - leg.distance_m
                    mask_mat[ia][ib] = int(leg.collected)

            dur[slot] = d_mat
            dist[slot] = dist_mat
            narrow[slot] = nar_mat
            masks[slot] = mask_mat

        return cls(
            point_ids=point_ids,
            poi_ids=poi_ids,
            poi_bits=poi_bits,
            poi_scores=poi_scores,
            snapped_nodes=snapped_nodes,
            node_poi_masks=node_poi_masks,
            dur=dur,
            dist=dist,
            narrow=narrow,
            masks=masks,
        )


class WaypointBusEnv(gym.Env):
    """Môi trường Gymnasium chọn POI kế tiếp (waypoint) trên ``WaypointTable``."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        table: WaypointTable,
        *,
        time_penalty: float = 0.005,
        narrow_penalty: float = 0.5,
        success_bonus: float = 1.0,
        fail_penalty: float = 5.0,
    ) -> None:
        super().__init__()
        self.table = table
        self.n_poi = len(table.poi_ids)
        self.finish_action = self.n_poi
        self.max_steps = self.n_poi + 1
        self.time_penalty = time_penalty
        self.narrow_penalty = narrow_penalty
        self.success_bonus = success_bonus
        self.fail_penalty = fail_penalty

        obs_dim = self.n_poi * WAYPOINT_CAND_FEATURES + WAYPOINT_GLOBAL_FEATURES
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_poi + 1)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        rng, t = self.np_random, self.table
        opt = options or {}

        if opt.get("start") is not None and opt.get("end") is not None:
            start_id = str(opt["start"])
            end_id = str(opt["end"])
            if start_id not in t.point_index or end_id not in t.point_index or start_id == end_id:
                raise ValueError("Choose distinct available endpoints")
        else:
            pairs = [
                (a, b)
                for a in t.point_ids
                for b in t.point_ids
                if a != b
                and t.snapped_nodes[a] != t.snapped_nodes[b]
                and math.isfinite(
                    float(t.dur["morning"][t.point_index[a], t.point_index[b]])
                )
            ]
            if not pairs:
                raise ValueError("No reachable endpoint pairs in WaypointTable")
            start_id, end_id = pairs[int(rng.integers(len(pairs)))]

        self.start_id = start_id
        self.end_id = end_id
        self.start_idx = t.point_index[start_id]
        self.end_idx = t.point_index[end_id]
        self.start_minute = int(opt.get("start_minute", rng.integers(6 * 60, 20 * 60)))
        slot0 = slot_of(self.start_minute)
        direct_s = float(t.dur[slot0][self.start_idx, self.end_idx])
        if not math.isfinite(direct_s):
            direct_s = float(t.dur["morning"][self.start_idx, self.end_idx])

        if "budget_s" in opt and opt["budget_s"] is not None:
            self.budget = float(opt["budget_s"])
        else:
            factor = float(rng.uniform(1.3, 6.0))
            extra_s = float(rng.uniform(10.0, 60.0) * 60.0)
            sampled = direct_s * factor + extra_s
            min_budget = max(direct_s * 1.15, 15.0 * 60.0)
            self.budget = float(np.clip(sampled, min_budget, 240.0 * 60.0))

        self.cur_id = start_id
        self.cur_idx = self.start_idx
        self.elapsed = 0.0
        self.distance_m = 0.0
        self.narrow_m = 0.0
        self.steps = 0
        self.collected_mask = int(t.node_poi_masks[start_id])
        self.score = t.mask_score(self.collected_mask)
        self.route = [start_id]
        return self._observe(), {
            "start": start_id,
            "end": end_id,
            "start_name": start_id,
            "end_name": end_id,
            "budget_min": self.budget / 60.0,
        }

    def _slot(self) -> str:
        return slot_of((self.start_minute + int(self.elapsed // 60)) % 1440)

    def _observe(self) -> np.ndarray:
        t = self.table
        slot = self._slot()
        dur_row = t.dur[slot][self.cur_idx]
        dur_col_end = t.dur[slot][:, self.end_idx]
        nar_row = t.narrow[slot][self.cur_idx]
        dist_row = t.dist[slot][self.cur_idx]
        mask_row = t.masks[slot][self.cur_idx]

        remain = max(self.budget - self.elapsed, 0.0)
        direct_back = float(dur_row[self.end_idx])
        end_node = t.snapped_nodes[self.end_id]

        feat = np.zeros((self.n_poi, WAYPOINT_CAND_FEATURES), dtype=np.float32)
        valid_mask = np.zeros(self.n_poi + 1, dtype=bool)
        max_dt_scale = max(self.budget, 60.0)

        for i, pid in enumerate(t.poi_ids):
            pidx = t.point_index[pid]
            bit = t.poi_bits[i]
            already = bool(self.collected_mask & (1 << bit))
            t_to = float(dur_row[pidx])
            t_back = float(dur_col_end[pidx])
            feasible = (
                (not already)
                and pid != self.end_id
                and pid != self.cur_id
                and t.snapped_nodes[pid] != end_node
                and math.isfinite(t_to)
                and math.isfinite(t_back)
                and (self.elapsed + t_to + t_back <= self.budget + 1e-6)
            )
            valid_mask[i] = feasible
            leg_mask = mask_row[pidx]
            new_bits = leg_mask & ~self.collected_mask
            marginal = t.mask_score(new_bits) if math.isfinite(t_to) else 0.0
            extra_s = (
                max(t_to + t_back - direct_back, 1.0)
                if math.isfinite(t_to) and math.isfinite(t_back) and math.isfinite(direct_back)
                else max_dt_scale
            )
            d_m = float(dist_row[pidx])
            n_m = float(nar_row[pidx])

            feat[i, 0] = 1.0 if feasible else 0.0
            feat[i, 1] = 1.0 if already else 0.0
            feat[i, 2] = np.clip(marginal / 10.0, 0.0, 5.0)
            feat[i, 3] = np.clip(t_to / max_dt_scale, 0.0, 2.0) if math.isfinite(t_to) else 2.0
            feat[i, 4] = np.clip(t_back / max_dt_scale, 0.0, 2.0) if math.isfinite(t_back) else 2.0
            feat[i, 5] = np.clip(extra_s / max_dt_scale, 0.0, 2.0)
            feat[i, 6] = np.clip((marginal * 60.0) / max(extra_s, 1.0), 0.0, 5.0)
            feat[i, 7] = (n_m / d_m) if d_m > 0 else 0.0

        # Hành động kết thúc (về thẳng end): luôn mở nếu còn đường hợp lệ, hoặc làm fallback an toàn.
        can_finish = math.isfinite(direct_back) and (
            self.elapsed + direct_back <= self.budget + 1e-6
        )
        valid_mask[self.finish_action] = can_finish or (not valid_mask[: self.n_poi].any())

        tod = 2.0 * math.pi * ((self.start_minute + self.elapsed / 60.0) % 1440) / 1440.0
        collected_count = sum(
            1 for bit in t.poi_bits if self.collected_mask & (1 << bit)
        )
        glob = np.array(
            [
                remain / max(self.budget, 1e-6),
                np.clip((remain - (direct_back if math.isfinite(direct_back) else self.budget)) / max(self.budget, 1e-6), 0.0, 1.0),
                np.clip((direct_back if math.isfinite(direct_back) else self.budget) / max(self.budget, 1e-6), 0.0, 1.0),
                math.sin(tod),
                math.cos(tod),
                collected_count / max(self.n_poi, 1),
                float(valid_mask[: self.n_poi].sum()) / max(self.n_poi, 1),
                self.steps / max(self.max_steps, 1),
            ],
            dtype=np.float32,
        )
        self._mask = valid_mask
        return np.concatenate([feat.ravel(), glob])

    def action_masks(self) -> np.ndarray:
        """Trả về action mask hợp lệ cho ``MaskablePPO``."""
        return self._mask

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        if action < 0 or action >= len(self._mask) or not self._mask[action]:
            raise ValueError(f"Invalid waypoint action: {action}")

        t = self.table
        slot = self._slot()
        next_id = self.end_id if action == self.finish_action else t.poi_ids[action]
        next_idx = t.point_index[next_id]

        dt = float(t.dur[slot][self.cur_idx, next_idx])
        d_m = float(t.dist[slot][self.cur_idx, next_idx])
        n_m = float(t.narrow[slot][self.cur_idx, next_idx])
        leg_mask = t.masks[slot][self.cur_idx][next_idx]

        if not math.isfinite(dt):
            self.steps += 1
            reward = -self.fail_penalty
            info = self._terminal_info(success=False)
            return self._observe(), float(reward), True, False, info

        new_bits = leg_mask & ~self.collected_mask
        gained = t.mask_score(new_bits)
        self.collected_mask |= leg_mask
        self.score = t.mask_score(self.collected_mask)
        self.elapsed += dt
        self.distance_m += d_m
        self.narrow_m += n_m
        self.steps += 1
        self.cur_id = next_id
        self.cur_idx = next_idx
        self.route.append(next_id)

        reward = (
            gained
            - self.time_penalty * (dt / 60.0)
            - self.narrow_penalty * (n_m / 1000.0)
        )
        arrived = action == self.finish_action
        late = self.elapsed > self.budget + 1e-6
        terminated = arrived or late
        truncated = (not terminated) and (self.steps >= self.max_steps)
        success = arrived and not late
        if success:
            reward += self.success_bonus
        elif terminated or truncated:
            reward -= self.fail_penalty

        info: dict[str, Any] = {
            "step": self.steps,
            "elapsed_min": self.elapsed / 60.0,
            "budget_min": self.budget / 60.0,
            "gained_score": gained,
            "narrow_m": self.narrow_m,
            "width_reward": -self.narrow_penalty * (n_m / 1000.0),
            "road_name": f"-> {next_id}",
            "from_label": self.route[-2],
            "to_label": next_id,
            "length_m": d_m,
            "width_m": None,
            "narrow": n_m > 0,
            "dt_s": dt,
            "valid_actions": int(self._mask.sum()),
            "degree": len(self._mask),
            "action": action,
        }
        if terminated or truncated:
            info.update(self._terminal_info(success=success))
        return self._observe(), float(reward), terminated, truncated, info

    def _terminal_info(self, *, success: bool) -> dict[str, Any]:
        collected_count = sum(
            1 for bit in self.table.poi_bits if self.collected_mask & (1 << bit)
        )
        return {
            "score": float(self.score),
            "success": float(success),
            "pois": int(collected_count),
            "start": self.start_id,
            "end": self.end_id,
            "start_name": self.start_id,
            "end_name": self.end_id,
            "narrow_m": float(self.narrow_m),
            "elapsed_min": float(self.elapsed / 60.0),
            "budget_min": float(self.budget / 60.0),
            "route": list(self.route),
        }


def train_waypoint(
    table: WaypointTable,
    epochs: int = 80,
    episodes_per_epoch: int = 64,
    update_passes: int = 6,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    hidden: int = 128,
    seed: int = 0,
    checkpoint: Path | None = CHECKPOINT_WAYPOINT,
    verbose: bool = False,
    log: Callable[[str], None] = print,
    **env_kwargs: Any,
) -> MaskablePPO:
    """Huấn luyện PPO ở mức waypoint bằng vòng lặp thu lô episode + GAE + PPO-clip."""
    env = WaypointBusEnv(table, **env_kwargs)
    model = MaskablePPO(
        "MlpPolicy",
        DummyVecEnv([lambda: WaypointBusEnv(table, **env_kwargs)]),
        learning_rate=learning_rate,
        policy_kwargs={"net_arch": {"pi": [hidden, hidden], "vf": [hidden, hidden]}},
        seed=seed,
        device="cpu",
    )
    rng = np.random.default_rng(seed)
    total_steps = 0

    for epoch in range(1, epochs + 1):
        episodes = []
        for _ in range(episodes_per_epoch):
            ep = run_episode(
                env,
                model.policy,
                seed=int(rng.integers(2**31)),
                verbose=verbose,
                log=log,
            )
            episodes.append(ep)
        parts: dict[str, list[np.ndarray]] = {
            k: [] for k in ("obs", "act", "mask", "logp", "adv", "ret")
        }
        for ep in episodes:
            adv, ret = compute_gae(
                ep["reward"], ep["value"], ep["last_value"], gamma, gae_lambda
            )
            for k, v in (
                ("obs", ep["obs"]),
                ("act", ep["act"]),
                ("mask", ep["mask"]),
                ("logp", ep["logp"]),
                ("adv", adv),
                ("ret", ret),
            ):
                parts[k].append(np.asarray(v))
        batch = {k: np.concatenate(v) for k, v in parts.items()}
        total_steps += len(batch["act"])
        stats = update_policy(
            model,
            batch,
            update_passes,
            batch_size,
            clip_range,
            ent_coef,
            vf_coef,
            max_grad_norm,
        )
        log(
            f"[epoch {epoch:3d}/{epochs}] steps={total_steps} | "
            f"mean_score={np.mean([e['info']['score'] for e in episodes]):.2f} "
            f"mean_pois={np.mean([e['info']['pois'] for e in episodes]):.2f} "
            f"success={np.mean([e['info']['success'] for e in episodes]):.2f} "
            f"pi_loss={stats['policy_loss']:.3f} vf_loss={stats['value_loss']:.3f}"
        )

    if checkpoint is not None:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        model.save(checkpoint)
    return model


def _pick_wp_random(
    _obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator, _model: Any
) -> int:
    return int(rng.choice(np.flatnonzero(mask)))


def _pick_wp_shortest(
    _obs: np.ndarray, mask: np.ndarray, _rng: np.random.Generator, _model: Any
) -> int:
    if mask[-1]:
        return len(mask) - 1
    return int(np.flatnonzero(mask)[0])


def _pick_wp_greedy(
    obs: np.ndarray, mask: np.ndarray, _rng: np.random.Generator, _model: Any
) -> int:
    n_poi = len(mask) - 1
    f = obs[: n_poi * WAYPOINT_CAND_FEATURES].reshape(n_poi, WAYPOINT_CAND_FEATURES)
    if not mask[:n_poi].any():
        return n_poi
    # Ưu tiên hiệu quả điểm/thời gian và tránh đường hẹp.
    scores = f[:, 6] + 2.0 * f[:, 2] - 0.5 * f[:, 7]
    valid = mask[:n_poi]
    return int(np.argmax(np.where(valid, scores, -np.inf)))


def _pick_wp_ppo(
    obs: np.ndarray, mask: np.ndarray, _rng: np.random.Generator, model: MaskablePPO
) -> int:
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    return int(action)


WAYPOINT_POLICIES = {
    "random": _pick_wp_random,
    "shortest": _pick_wp_shortest,
    "greedy": _pick_wp_greedy,
    "ppo": _pick_wp_ppo,
}


def evaluate_waypoint(
    table: WaypointTable,
    policy: str,
    model: MaskablePPO | None = None,
    episodes: int = 30,
    seed: int = 1000,
    **env_kwargs: Any,
) -> dict[str, Any]:
    """Đánh giá chuẩn các chính sách waypoint trên cùng ``episodes`` kịch bản cố định."""
    pick, env = WAYPOINT_POLICIES[policy], WaypointBusEnv(table, **env_kwargs)
    rng = np.random.default_rng(seed)
    scores, wins, pois, narrows = [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, info = False, {}
        while not done:
            obs, _, term, trunc, info = env.step(
                pick(obs, env.action_masks(), rng, model)
            )
            done = term or trunc
        scores.append(info["score"])
        wins.append(info["success"])
        pois.append(info["pois"])
        narrows.append(info["narrow_m"])
    return {
        "policy": policy,
        "mean_score": float(np.mean(scores)),
        "mean_pois": float(np.mean(pois)),
        "mean_narrow_m": float(np.mean(narrows)),
        "success_rate": float(np.mean(wins)),
    }


class PoiPPOSolver(PoiBaseSolver):
    """Bộ giải ``MaskablePPO`` ở mức waypoint kế thừa ``PoiBaseSolver``."""

    name = "ppo"

    def __init__(
        self, checkpoint: str | Path, *, model: MaskablePPO | None = None
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.model = model

    @staticmethod
    def _problem_observation(
        problem: RoadProblem,
        poi_ids: list[str],
        poi_bits: list[int],
        cur_id: str,
        elapsed: float,
        collected_mask: int,
        steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dựng vector quan sát và action mask trực tiếp từ ``RoadProblem`` (bảo đảm quy tắc đích sink)."""
        graph = problem.graph
        n_poi = len(poi_ids)
        finish_action = n_poi
        max_steps = n_poi + 1
        remain = max(problem.budget_s - elapsed, 0.0)
        max_dt_scale = max(problem.budget_s, 60.0)

        direct_conn = problem.timed_connection(cur_id, problem.end, elapsed)
        direct_back = (
            direct_conn[1] - elapsed if direct_conn is not None else float("inf")
        )

        feat = np.zeros((n_poi, WAYPOINT_CAND_FEATURES), dtype=np.float32)
        valid_mask = np.zeros(n_poi + 1, dtype=bool)
        end_node = graph.snapped[problem.end]

        for i, pid in enumerate(poi_ids):
            bit = poi_bits[i]
            already = bool(collected_mask & (1 << bit))
            feasible = False
            t_to = float("inf")
            t_back = float("inf")
            marginal = 0.0
            d_m = 0.0
            n_m = 0.0

            if (
                not already
                and pid != problem.end
                and pid != cur_id
                and pid in graph.snapped
                and graph.snapped[pid] != end_node
            ):
                conn_to = problem.timed_connection(cur_id, pid, elapsed)
                if conn_to is not None:
                    leg_to, arr_to, _ = conn_to
                    t_to = arr_to - elapsed
                    conn_back = problem.timed_connection(pid, problem.end, arr_to)
                    if conn_back is not None:
                        _, arr_end, _ = conn_back
                        t_back = arr_end - arr_to
                        feasible = arr_end <= problem.budget_s + 1e-7
                    new_bits = leg_to.collected & ~collected_mask
                    marginal = graph.mask_score(new_bits)
                    d_m = leg_to.distance_m
                    n_m = leg_to.weighted_m - leg_to.distance_m

            valid_mask[i] = feasible
            extra_s = (
                max(t_to + t_back - direct_back, 1.0)
                if math.isfinite(t_to) and math.isfinite(t_back) and math.isfinite(direct_back)
                else max_dt_scale
            )
            feat[i, 0] = 1.0 if feasible else 0.0
            feat[i, 1] = 1.0 if already else 0.0
            feat[i, 2] = np.clip(marginal / 10.0, 0.0, 5.0)
            feat[i, 3] = np.clip(t_to / max_dt_scale, 0.0, 2.0) if math.isfinite(t_to) else 2.0
            feat[i, 4] = np.clip(t_back / max_dt_scale, 0.0, 2.0) if math.isfinite(t_back) else 2.0
            feat[i, 5] = np.clip(extra_s / max_dt_scale, 0.0, 2.0)
            feat[i, 6] = np.clip((marginal * 60.0) / max(extra_s, 1.0), 0.0, 5.0)
            feat[i, 7] = (n_m / d_m) if d_m > 0 else 0.0

        can_finish = math.isfinite(direct_back) and (
            elapsed + direct_back <= problem.budget_s + 1e-7
        )
        valid_mask[finish_action] = can_finish or (not valid_mask[:n_poi].any())

        tod = 2.0 * math.pi * ((problem.start_minute + elapsed / 60.0) % 1440) / 1440.0
        collected_count = sum(1 for bit in poi_bits if collected_mask & (1 << bit))
        glob = np.array(
            [
                remain / max(problem.budget_s, 1e-6),
                np.clip(
                    (remain - (direct_back if math.isfinite(direct_back) else problem.budget_s))
                    / max(problem.budget_s, 1e-6),
                    0.0,
                    1.0,
                ),
                np.clip(
                    (direct_back if math.isfinite(direct_back) else problem.budget_s)
                    / max(problem.budget_s, 1e-6),
                    0.0,
                    1.0,
                ),
                math.sin(tod),
                math.cos(tod),
                collected_count / max(n_poi, 1),
                float(valid_mask[:n_poi].sum()) / max(n_poi, 1),
                steps / max(max_steps, 1),
            ],
            dtype=np.float32,
        )
        return np.concatenate([feat.ravel(), glob]), valid_mask

    def solve(self, context: PoiContext, request: RouteRequest) -> RoadResult:
        road_graph = context.road_graph
        problem = context.problem

        poi_pairs = [
            (bit, pid)
            for bit, pid in enumerate(road_graph.pois)
            if pid in road_graph.snapped
        ]
        poi_ids = [pid for _, pid in poi_pairs]
        poi_bits = [bit for bit, _ in poi_pairs]
        n_poi = len(poi_ids)

        model = self.model
        if model is None:
            if not self.checkpoint.is_file():
                raise FileNotFoundError(f"Waypoint PPO checkpoint not found: {self.checkpoint}")
            model = MaskablePPO.load(self.checkpoint, device="cpu")

        expected_obs = (n_poi * WAYPOINT_CAND_FEATURES + WAYPOINT_GLOBAL_FEATURES,)
        if tuple(model.observation_space.shape or ()) != expected_obs:
            raise ValueError(
                f"Checkpoint '{self.checkpoint}' has observation shape "
                f"{model.observation_space.shape}, expected {expected_obs}"
            )
        if getattr(model.action_space, "n", None) != n_poi + 1:
            raise ValueError(
                f"Checkpoint '{self.checkpoint}' has action space {model.action_space}, "
                f"expected Discrete({n_poi + 1})"
            )

        cur_id = request.start
        elapsed = 0.0
        collected_mask = int(road_graph.node_mask(road_graph.snapped[request.start]))
        route: list[str] = [request.start]
        steps = 0

        while steps < n_poi + 1:
            request.check_cancelled()
            obs, mask = self._problem_observation(
                problem, poi_ids, poi_bits, cur_id, elapsed, collected_mask, steps
            )
            action_arr, _ = model.predict(obs, action_masks=mask, deterministic=True)
            action = int(action_arr)
            if action == n_poi:
                route.append(request.end)
                break
            nxt_id = poi_ids[action]
            conn = problem.timed_connection(cur_id, nxt_id, elapsed)
            if conn is None:
                route.append(request.end)
                break
            leg, elapsed, _ = conn
            collected_mask |= leg.collected
            cur_id = nxt_id
            route.append(nxt_id)
            steps += 1
        else:
            if route[-1] != request.end:
                route.append(request.end)

        result = problem.evaluate(route)
        return result if result is not None else problem.direct


_problem_observation = PoiPPOSolver._problem_observation


def infer_waypoint_route(
    nodes_path: str | Path,
    edges_path: str | Path,
    widths_path: str | Path,
    points: Sequence[dict[str, Any]],
    start: str,
    end: str,
    checkpoint: str | Path,
    *,
    budget_min: float = 240.0,
    start_minute: int = 480,
    cancelled: Callable[[], bool] = lambda: False,
    reduced_dir: Path = REDUCED_DIR,
    use_reduced: bool = True,
    workers: int = 1,
    road_graph: RoadGraph | None = None,
    problem: RoadProblem | None = None,
    model: MaskablePPO | None = None,
) -> dict[str, Any]:
    """Suy luận lộ trình waypoint bằng PPO và đánh giá/vẽ hoàn toàn qua ``RoadProblem``."""
    request = RouteRequest(
        start=start,
        end=end,
        budget_min=budget_min,
        start_minute=start_minute,
        workers=workers,
        cancelled=cancelled,
    )
    solver = PoiPPOSolver(checkpoint, model=model)
    context = None
    if road_graph is not None and problem is not None:
        context = PoiContext(road_graph=road_graph, problem=problem)
    elif road_graph is not None:
        context = PoiContext(
            road_graph=road_graph,
            problem=RoadProblem(
                road_graph,
                start,
                end,
                budget_min,
                start_minute=start_minute,
                all_pois=True,
                workers=workers,
            ),
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
