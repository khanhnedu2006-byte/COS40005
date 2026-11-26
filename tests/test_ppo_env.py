"""Kiểm thử môi trường PPO và một vòng huấn luyện nhỏ trên đồ thị đã rút gọn."""

import numpy as np
import pytest

from busroute.routing.ppo_env import NARROW_WIDTH_M, REDUCED_DIR, BusRouteEnv, GraphData

pytestmark = pytest.mark.skipif(
    not (REDUCED_DIR / "nodes_reduced.parquet").exists(),
    reason="chạy scripts/reduce_graph.py trước",
)


@pytest.fixture(scope="module")
def graph():
    """Nạp một lần ``GraphData`` cho toàn bộ module kiểm thử."""
    return GraphData.load()


def test_episode_always_ends_at_target_in_budget(graph):
    """Mọi episode đều phải tới đích B trong ngân sách nhờ action mask."""
    env = BusRouteEnv(graph)
    rng = np.random.default_rng(0)
    for seed in range(3):
        env.reset(seed=seed)
        done, info = False, {}
        while not done:
            mask = env.action_masks()
            assert mask.any()  # mask không bao giờ rỗng
            _, _, term, trunc, info = env.step(int(rng.choice(np.flatnonzero(mask))))
            done = term or trunc
        assert info["success"] == 1.0  # che cạnh không kịp -> luôn về được B đúng hạn
        assert env.elapsed <= env.budget + 1e-6


def test_invalid_action_rejected(graph):
    """Chọn cạnh bị che trong action mask phải ném ValueError."""
    env = BusRouteEnv(graph)
    env.reset(seed=0)
    bad = int(np.flatnonzero(~env.action_masks())[0]) if (~env.action_masks()).any() else env.K
    with pytest.raises(ValueError):
        env.step(bad)


def test_poi_to_poi_names_in_info(graph):
    """Điểm đầu/cuối mặc định là hai POI khác nút nhau và trả về đúng tên POI."""
    env = BusRouteEnv(graph)
    _, reset_info = env.reset(seed=42)
    assert reset_info["start"] != reset_info["end"]
    assert reset_info["start_name"] in graph.poi_names
    assert reset_info["end_name"] in graph.poi_names

    _, named = env.reset(seed=0, options={"start": "P001", "end": "P003"})
    assert named["start_name"] == graph.poi_names[graph.poi_by_id["P001"]]
    assert named["end_name"] == graph.poi_names[graph.poi_by_id["P003"]]


def test_road_width_reward_penalizes_narrow_and_unknown(graph):
    """Đường >= 5.5 m không bị phạt; đường chưa rõ bị phạt nhẹ; đường hẹp bị phạt nặng."""
    env = BusRouteEnv(graph)
    narrow_idx = int(np.flatnonzero(graph.narrow)[0])
    wide_idx = int(np.flatnonzero(np.isfinite(graph.width) & (~graph.narrow))[0])
    unknown_idx = int(np.flatnonzero(~np.isfinite(graph.width))[0])

    assert env._width_reward(wide_idx) == 0.0
    assert env._width_reward(unknown_idx) < 0.0
    assert env._width_reward(narrow_idx) < 0.0

    # So sánh mức phạt trên mỗi km: đường hẹp phải nặng hơn đường chưa rõ bề rộng.
    narrow_per_km = env._width_reward(narrow_idx) / (graph.length[narrow_idx] / 1000.0)
    unknown_per_km = env._width_reward(unknown_idx) / (graph.length[unknown_idx] / 1000.0)
    assert narrow_per_km < unknown_per_km <= -env.unknown_width_penalty + 1e-9
    assert graph.width[narrow_idx] < NARROW_WIDTH_M


def test_ppo_smoke(graph, tmp_path):
    """Chạy thử 2 epoch nhỏ để kiểm tra toàn bộ chu trình train + evaluate."""
    from busroute.routing.ppo_train import evaluate, train

    model = train(
        graph,
        epochs=2,
        episodes_per_epoch=2,
        batch_size=64,
        update_passes=1,
        hidden=16,
        checkpoint=tmp_path / "m.zip",
        log=lambda *_: None,
    )
    row = evaluate(graph, "ppo", model, episodes=2)
    assert 0.0 <= row["success_rate"] <= 1.0


def test_verbose_logs_every_step(graph, tmp_path):
    """Cờ verbose phải ghi log đúng từng step và hiển thị tên POI đầu/cuối."""
    from busroute.routing.ppo_train import run_episode, train

    model = train(
        graph,
        epochs=1,
        episodes_per_epoch=1,
        batch_size=16,
        update_passes=1,
        hidden=8,
        max_episode_steps=15,
        checkpoint=tmp_path / "v.zip",
        log=lambda *_: None,
    )
    lines: list[str] = []
    env = BusRouteEnv(graph, max_episode_steps=15)
    ep = run_episode(env, model.policy, seed=0, verbose=True, log=lines.append)

    step_lines = [line for line in lines if line.startswith("    step ")]
    assert lines[0].startswith("  [start] ")
    assert "node " not in lines[0]
    assert len(step_lines) == ep["info"]["steps"]


def test_gae_matches_hand_computation():
    """Kiểm tra công thức GAE bằng tính tay trên 2 step."""
    from busroute.routing.ppo_train import compute_gae

    # 2 step, gamma=1, lam=1: A_1 = r_1 - V_1 ; A_0 = r_0 + V_1 - V_0 + A_1
    adv, ret = compute_gae([1.0, 2.0], [0.5, 1.0], 0.0, 1.0, 1.0)
    assert np.allclose(adv, [2.5, 1.0])
    assert np.allclose(ret, [3.0, 2.0])
