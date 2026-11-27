"""Huấn luyện PPO THEO EPISODE (thu cả lô episode rồi mới học một lượt) + đánh giá.

Phân cấp (từ lớn tới nhỏ):  EPOCH  >  EPISODE  >  STEP.

  * STEP    : một quyết định chọn cạnh (đơn vị nhỏ nhất, có reward).
  * EPISODE : một chuyến POI A -> POI B trọn vẹn, gồm nhiều step.
  * EPOCH   : một vòng huấn luyện, gồm nhiều episode; cuối epoch mạng được học một lượt.

Mỗi EPOCH gồm đúng ba pha:

  1. THU THẬP: cho agent chạy ``episodes_per_epoch`` episode TRỌN VẸN (mỗi episode là
     một chuyến A -> B gồm nhiều step). Trong pha này mạng KHÔNG được cập nhật; chính
     sách giữ nguyên để dữ liệu nhất quán. Mỗi step lưu: quan sát, hành động, mask,
     log-xác-suất lúc chọn, value dự đoán, phần thưởng.
  2. TÍNH LỢI THẾ: với từng episode, đi NGƯỢC từ step cuối về step đầu để tính return
     và lợi thế GAE. Nhờ vậy điểm POI nhận ở cuối chuyến được "chia công" cho các
     quyết định trước đó.
  3. HỌC MỘT LƯỢT: gộp toàn bộ step của cả lô, xáo trộn, chia minibatch và tối ưu
     ``update_passes`` lượt duyệt qua lô theo hàm mục tiêu PPO-clip (không cho chính sách
     đổi quá xa chính sách đã dùng để thu dữ liệu). Xong thì bỏ lô, quay lại pha 1.

Mạng chính sách/giá trị là của sb3-contrib (``MaskablePPO.policy``) nên vẫn lưu/nạp và
``predict`` như SB3; phần vòng lặp thu-thập/GAE/cập nhật được viết tường minh ở đây.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from busroute.graph.ab_estimator import PARAMS_FILE
from busroute.routing.base import RoadDatasetSpec, RouteRequest
from busroute.routing.edge_metaheuristics import EdgeWalk, StreetBaseSolver, StreetContext
from busroute.routing.ppo_env import (
    EDGE_FEATURES,
    GLOBAL_FEATURES,
    REDUCED_DIR,
    BusRouteEnv,
    GraphData,
)
from busroute.routing.road_dijkstra import road_widths
from busroute.routing.road_graph import NoRoadRouteError
from busroute.utils.timeslots import slot_of

CHECKPOINT = Path(__file__).resolve().parents[3] / "checkpoints" / "ppo_reduced.zip"
CHECKPOINT_FULL = Path(__file__).resolve().parents[3] / "checkpoints" / "ppo_full.zip"


def run_episode(
    env: BusRouteEnv,
    policy: Any,
    seed: int | None = None,
    verbose: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Chạy MỘT episode trọn vẹn với chính sách hiện tại và trả về toàn bộ quỹ đạo.

    Không cập nhật mạng ở đây. Kết thúc episode có hai loại:
      * ``terminated``: tới B (hoặc quá hạn) -> không còn tương lai, value kế tiếp = 0;
      * ``truncated``: hết số step tối đa -> episode bị cắt giữa chừng, nên khi tính
        return phải "mồi" (bootstrap) bằng value của trạng thái cuối.

    Args:
        env: Môi trường ``BusRouteEnv`` để chạy chuyến xe.
        policy: Chính sách actor-critic (``MaskablePPO.policy``) có ``forward`` và
            ``predict_values``.
        seed: Seed tùy chọn cho ``env.reset()`` để tái lập kịch bản A -> B.
        verbose: Nếu ``True``, in từng step (nút, tên đường, bề rộng, phần thưởng).
        log: Hàm ghi log nhận một chuỗi (mặc định ``print``).

    Returns:
        Dictionary chứa danh sách ``obs``, ``act``, ``mask``, ``logp``, ``value``,
        ``reward``, cùng ``last_value`` (giá trị mồi) và ``info`` tổng kết episode.
    """
    obs, reset_info = env.reset(seed=seed)
    if verbose:
        log(
            f"  [start] {reset_info['start_name']} -> {reset_info['end_name']} "
            f"(budget={reset_info['budget_min']:.1f} min)"
        )
    traj: dict[str, Any] = {k: [] for k in ("obs", "act", "mask", "logp", "value", "reward")}
    done = truncated = False
    info: dict[str, Any] = {}
    while not done:
        mask = env.action_masks()
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, value, logp = policy(tensor, action_masks=mask[None])
        a = int(action.item())
        next_obs, reward, terminated, truncated, info = env.step(a)
        if verbose:
            w_str = (
                f"{info['width_m']:.1f}m{' [NARROW]' if info['narrow'] else ''}"
                if info["width_m"] is not None
                else "unknown"
            )
            log(
                f"    step {info['step']:4d}: {info['from_label']} -> {info['to_label']} | "
                f"road={info['road_name']} ({info['length_m']:.0f}m, w={w_str}) | "
                f"dt={info['dt_s']:.1f}s | r={reward:+.3f} "
                f"(poi={info['gained_score']:+.2f}, width={info['width_reward']:+.3f}) | "
                f"{info['elapsed_min']:.1f}/{info['budget_min']:.1f}m | "
                f"act={a} ({info['valid_actions']}/{info['degree']} valid)"
            )
        for k, v in zip(traj, (obs, a, mask, logp.item(), value.item(), reward)):
            traj[k].append(v)
        obs, done = next_obs, terminated or truncated
    # Giá trị mồi: chỉ khác 0 khi episode bị cắt (truncated), không phải kết thúc thật.
    last_value = 0.0
    if truncated:
        with torch.no_grad():
            last_value = policy.predict_values(
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            ).item()
    traj["last_value"] = last_value
    traj["info"] = {**info, "steps": len(traj["reward"]), "return": float(sum(traj["reward"]))}
    return traj


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Tính lợi thế GAE và return mục tiêu cho MỘT episode, duyệt ngược từ step cuối.

    Công thức:
      ``delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)``
      ``A_t     = delta_t + gamma * lam * A_{t+1}``
      ``return_t = A_t + V(s_t)``

    Args:
        rewards: Chuỗi phần thưởng tại từng step của episode.
        values: Chuỗi giá trị dự đoán ``V(s_t)`` tại từng step.
        last_value: Giá trị mồi ``V(s_T)`` nếu bị cắt, hoặc ``0.0`` nếu kết thúc thật.
        gamma: Hệ số chiết khấu tương lai trong khoảng ``[0, 1]``.
        lam: Tham số GAE lambda điều chỉnh cân bằng bias/variance.

    Returns:
        Bộ ``(advantages, returns)`` đều là mảng ``float32`` cùng độ dài ``rewards``.
    """
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    next_value, next_adv = last_value, 0.0
    for t in reversed(range(n)):
        delta = rewards[t] + gamma * next_value - values[t]
        next_adv = delta + gamma * lam * next_adv
        adv[t] = next_adv
        next_value = values[t]
    return adv, adv + np.asarray(values, dtype=np.float32)


def update_policy(
    model: MaskablePPO,
    batch: dict[str, np.ndarray],
    update_passes: int,
    batch_size: int,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
) -> dict[str, float]:
    """Pha 3: học MỘT LƯỢT trên toàn bộ lô episode bằng hàm mục tiêu PPO-clip.

    Args:
        model: Thực thể ``MaskablePPO`` chứa mạng chính sách, value và optimizer.
        batch: Lô dữ liệu gộp từ các episode gồm ``obs``, ``act``, ``mask``,
            ``logp``, ``adv``, ``ret``.
        update_passes: Số lượt duyệt qua toàn bộ lô dữ liệu.
        batch_size: Kích thước mỗi minibatch khi cập nhật gradient.
        clip_range: Ngưỡng cắt tỉ số xác suất ``[1 - clip, 1 + clip]`` của PPO.
        ent_coef: Hệ số thưởng entropy khuyến khích khám phá.
        vf_coef: Trọng số hàm mất mát của value network.
        max_grad_norm: Ngưỡng cắt chuẩn gradient để tránh bùng nổ cập nhật.

    Returns:
        Dictionary chứa ``policy_loss`` và ``value_loss`` ở minibatch cuối cùng.
    """
    policy = model.policy
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
    act = torch.as_tensor(batch["act"], dtype=torch.long)
    masks = torch.as_tensor(batch["mask"], dtype=torch.bool)
    old_logp = torch.as_tensor(batch["logp"], dtype=torch.float32)
    adv = torch.as_tensor(batch["adv"], dtype=torch.float32)
    ret = torch.as_tensor(batch["ret"], dtype=torch.float32)
    # Chuẩn hoá lợi thế trên cả lô để thang đo ổn định.
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = len(act)
    stats: dict[str, float] = {}
    for _ in range(update_passes):
        for idx in torch.randperm(n).split(batch_size):
            value, logp, entropy = policy.evaluate_actions(obs[idx], act[idx], action_masks=masks[idx])
            # Tỉ số xác suất mới/cũ; cắt trong [1-clip, 1+clip] để cập nhật không quá mạnh.
            ratio = torch.exp(logp - old_logp[idx])
            surrogate = torch.min(
                adv[idx] * ratio, adv[idx] * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            )
            policy_loss = -surrogate.mean()
            value_loss = torch.nn.functional.mse_loss(value.flatten(), ret[idx])
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()
            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            policy.optimizer.step()
            stats = {"policy_loss": policy_loss.item(), "value_loss": value_loss.item()}
    return stats


def train(
    graph: GraphData,
    epochs: int = 100,
    episodes_per_epoch: int = 16,
    update_passes: int = 6,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    hidden: int = 128,
    seed: int = 0,
    checkpoint: Path | None = CHECKPOINT,
    verbose: bool = False,
    log: Callable[[str], None] = print,
    **env_kwargs: Any,
) -> MaskablePPO:
    """Huấn luyện PPO theo phân cấp epoch > episode > step và lưu checkpoint.

    ``gamma`` cao vì phần thưởng (POI) có thể đến sau hàng trăm step.

    Args:
        graph: Dữ liệu đồ thị rút gọn dùng chung.
        epochs: Số epoch huấn luyện.
        episodes_per_epoch: Số episode trọn vẹn thu thập trước mỗi lần học.
        update_passes: Số lượt duyệt qua lô dữ liệu trong mỗi lần cập nhật PPO.
        batch_size: Kích thước minibatch trong pha cập nhật gradient.
        learning_rate: Tốc độ học cho bộ tối ưu Adam của mạng actor-critic.
        gamma: Hệ số chiết khấu phần thưởng tương lai.
        gae_lambda: Tham số GAE lambda.
        clip_range: Ngưỡng cắt tỉ số xác suất PPO.
        ent_coef: Hệ số thưởng entropy.
        vf_coef: Trọng số mất mát value network.
        max_grad_norm: Ngưỡng cắt chuẩn gradient.
        hidden: Số nơ-ron mỗi tầng ẩn của MLP (hai tầng ẩn cho pi và vf).
        seed: Seed ngẫu nhiên cho huấn luyện và sinh kịch bản.
        checkpoint: Đường dẫn lưu mô hình ``.zip`` (``None`` để không lưu).
        verbose: Nếu ``True``, hiển thị chi tiết từng step của mỗi episode.
        log: Hàm ghi log nhận một chuỗi (mặc định ``print``).
        **env_kwargs: Tham số truyền thẳng vào ``BusRouteEnv``.

    Returns:
        Thực thể ``MaskablePPO`` đã huấn luyện xong.
    """
    env = BusRouteEnv(graph, **env_kwargs)
    # MaskablePPO chỉ dùng làm "vỏ" tạo mạng + optimizer + save/load; không gọi .learn().
    model = MaskablePPO(
        "MlpPolicy",
        DummyVecEnv([lambda: BusRouteEnv(graph, **env_kwargs)]),
        learning_rate=learning_rate,
        policy_kwargs={"net_arch": {"pi": [hidden, hidden], "vf": [hidden, hidden]}},
        seed=seed,
        device="cpu",
    )
    rng = np.random.default_rng(seed)
    total_steps = 0
    for epoch in range(1, epochs + 1):
        log(f"=== EPOCH {epoch}/{epochs} ===")
        # ---- Pha 1: thu thập các episode trọn vẹn (mạng đứng yên) ----
        episodes = []
        for k in range(1, episodes_per_epoch + 1):
            ep = run_episode(
                env,
                model.policy,
                seed=int(rng.integers(2**31)),
                verbose=verbose,
                log=log,
            )
            episodes.append(ep)
            i = ep["info"]
            log(
                f"  episode {k}/{episodes_per_epoch}: {i['start_name']} -> {i['end_name']}  "
                f"steps={i['steps']}  return={i['return']:+.2f}  poi_score={i['score']:.2f} "
                f"({i['pois']} POIs)  narrow={i['narrow_m']:.0f}m  "
                f"{i['elapsed_min']:.0f}/{i['budget_min']:.0f} min  "
                f"{'reached B' if i['success'] else 'FAILED'}"
            )
        # ---- Pha 2: GAE cho từng episode rồi gộp thành một lô ----
        parts: dict[str, list[np.ndarray]] = {
            k: [] for k in ("obs", "act", "mask", "logp", "adv", "ret")
        }
        for ep in episodes:
            adv, ret = compute_gae(ep["reward"], ep["value"], ep["last_value"], gamma, gae_lambda)
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
        # ---- Pha 3: học một lượt trên cả lô ----
        stats = update_policy(
            model, batch, update_passes, batch_size, clip_range, ent_coef, vf_coef, max_grad_norm
        )
        log(
            f"  -> learn: {len(episodes)} episodes, {len(batch['act'])} steps (total {total_steps}) | "
            f"mean poi_score={np.mean([e['info']['score'] for e in episodes]):.2f} "
            f"success={np.mean([e['info']['success'] for e in episodes]):.2f} "
            f"policy_loss={stats['policy_loss']:.3f} value_loss={stats['value_loss']:.3f}"
        )
    if checkpoint is not None:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        model.save(checkpoint)
    return model


# ---------------------------------------------------------------- baselines
def _pick_random(
    obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator, _model: Any
) -> int:
    """Chọn ngẫu nhiên một cạnh hợp lệ.

    Args:
        obs: Vector quan sát hiện tại (không dùng).
        mask: Mảng boolean action mask.
        rng: Bộ sinh số ngẫu nhiên NumPy.
        _model: Mô hình PPO (không dùng).

    Returns:
        Chỉ số hành động được chọn.
    """
    return int(rng.choice(np.flatnonzero(mask)))


def _pick_shortest(
    obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator, _model: Any
) -> int:
    """Đi thẳng về B theo đường ngắn nhất (không gom POI): cận dưới về điểm.

    Args:
        obs: Vector quan sát hiện tại.
        mask: Mảng boolean action mask.
        rng: Bộ sinh số ngẫu nhiên NumPy (không dùng).
        _model: Mô hình PPO (không dùng).

    Returns:
        Chỉ số hành động có tỉ lệ tiến tới đích B lớn nhất.
    """
    progress = obs[: len(mask) * EDGE_FEATURES].reshape(len(mask), EDGE_FEATURES)[:, 3]
    return int(np.argmax(np.where(mask, progress, -np.inf)))


def _pick_greedy(
    obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator, _model: Any
) -> int:
    """Tham lam: ưu tiên POI trước mặt, tiến gần POI trung gian, né đường hẹp và lặp nút.

    Args:
        obs: Vector quan sát hiện tại.
        mask: Mảng boolean action mask.
        rng: Bộ sinh số ngẫu nhiên NumPy (không dùng).
        _model: Mô hình PPO (không dùng).

    Returns:
        Chỉ số hành động đạt điểm heuristic cao nhất trong các cạnh hợp lệ.
    """
    f = obs[: len(mask) * EDGE_FEATURES].reshape(len(mask), EDGE_FEATURES)
    value = f[:, 5] * 10 + f[:, 4] + 0.01 * f[:, 3] - 0.1 * f[:, 6] - 0.2 * f[:, 7]
    return int(np.argmax(np.where(mask, value, -np.inf)))


def _pick_ppo(
    obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator, model: MaskablePPO
) -> int:
    """Suy luận tất định (deterministic) từ mô hình PPO đã huấn luyện.

    Args:
        obs: Vector quan sát hiện tại.
        mask: Mảng boolean action mask.
        rng: Bộ sinh số ngẫu nhiên NumPy (không dùng).
        model: Mô hình ``MaskablePPO``.

    Returns:
        Chỉ số hành động có xác suất cao nhất theo policy.
    """
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    return int(action)


POLICIES = {
    "random": _pick_random,
    "shortest": _pick_shortest,
    "greedy": _pick_greedy,
    "ppo": _pick_ppo,
}


def evaluate(
    graph: GraphData,
    policy: str,
    model: MaskablePPO | None = None,
    episodes: int = 30,
    seed: int = 1000,
    **env_kwargs: Any,
) -> dict[str, Any]:
    """Chạy ``episodes`` kịch bản cố định để so sánh các chính sách trên cùng tập POI A -> B.

    Args:
        graph: Dữ liệu đồ thị rút gọn dùng chung.
        policy: Tên chính sách (``"random"``, ``"shortest"``, ``"greedy"``, hoặc ``"ppo"``).
        model: Mô hình ``MaskablePPO`` (bắt buộc khi ``policy == "ppo"``).
        episodes: Số episode đánh giá cố định theo ``seed + ep``.
        seed: Seed gốc để sinh cùng bộ kịch bản A/B/giờ/ngân sách cho mọi chính sách.
        **env_kwargs: Tham số truyền thẳng vào ``BusRouteEnv``.

    Returns:
        Dictionary gồm ``policy``, ``mean_score``, ``mean_pois``, ``mean_narrow_m``
        và ``success_rate``.
    """
    pick, env = POLICIES[policy], BusRouteEnv(graph, **env_kwargs)
    rng = np.random.default_rng(seed)
    scores, wins, pois, narrows = [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, info = False, {}
        while not done:
            obs, _, term, trunc, info = env.step(pick(obs, env.action_masks(), rng, model))
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


class StreetPPOSolver(StreetBaseSolver):
    """Bộ giải ``MaskablePPO`` ở mức cạnh kế thừa ``StreetBaseSolver``."""

    name = "ppo"

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        model: MaskablePPO | None = None,
        max_steps: int = 1000,
    ) -> None:
        super().__init__(max_steps=max_steps)
        self.checkpoint = Path(checkpoint)
        self.model = model

    def solve(self, context: StreetContext, request: RouteRequest) -> EdgeWalk:
        graph = context.graph
        if request.start not in graph.point_by_id or request.end not in graph.point_by_id:
            raise ValueError("Start and end must be available road points within 300 m")

        start_node, _ = graph.point_by_id[request.start]
        end_node, _ = graph.point_by_id[request.end]
        direct = float(graph.to_target(slot_of(request.start_minute), end_node)[start_node])
        if not math.isfinite(direct) or direct > request.budget_s + 1e-6:
            raise NoRoadRouteError("No directed route found within the time budget")

        model = self.model
        if model is None:
            if not self.checkpoint.is_file():
                raise FileNotFoundError(f"PPO checkpoint not found: {self.checkpoint}")
            model = MaskablePPO.load(self.checkpoint, device="cpu")
        expected_shape = (graph.max_degree * EDGE_FEATURES + GLOBAL_FEATURES,)
        if tuple(model.observation_space.shape or ()) != expected_shape:
            raise ValueError(
                f"Checkpoint '{self.checkpoint}' has observation shape "
                f"{model.observation_space.shape}, expected {expected_shape}"
            )

        env = BusRouteEnv(graph)
        obs, _ = env.reset(
            seed=0,
            options={
                "start": request.start,
                "end": request.end,
                "start_minute": request.start_minute,
                "budget_s": request.budget_s,
            },
        )

        edges_taken: list[int] = []
        done = False
        info: dict[str, Any] = {}
        while not done:
            request.check_cancelled()
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            edges_taken.append(int(info["edge_idx"]))
            done = terminated or truncated

        if not info.get("success"):
            raise ValueError("PPO route did not reach the destination within the time budget")
        walk = self._evaluate_walk(
            graph,
            start_node,
            end_node,
            edges_taken,
            request.start_minute,
            request.budget_s,
        )
        if walk is None:
            raise ValueError("PPO route did not reach the destination within the time budget")
        return walk


def infer_route(
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
    graph: GraphData | None = None,
    model: MaskablePPO | None = None,
    orig_coords: np.ndarray | None = None,
) -> dict[str, Any]:
    """Suy luận một lộ trình hoàn chỉnh bằng checkpoint PPO và trả về ``RouteSnapshot`` cho Web."""
    request = RouteRequest(
        start=start,
        end=end,
        budget_min=budget_min,
        start_minute=start_minute,
        cancelled=cancelled,
    )
    solver = StreetPPOSolver(checkpoint, model=model)
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
