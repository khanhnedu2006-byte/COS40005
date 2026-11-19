"""Road PPO correctness, gradient flow, masking and checkpoint integrity."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stable_baselines3")
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from busroute.routing.ppo_road import (
    RoadEnv,
    RoadFeatures,
    RoadPPOConfig,
    RoadTrainingEnv,
    evaluate,
    fingerprint,
    infer,
    load_policy,
    make_model,
    read_checkpoint,
    sample_env,
    train,
)
from busroute.routing.road_graph import RoadGraph, RoadProblem


@pytest.fixture
def features() -> RoadFeatures:
    """Build a four-node test graph with a narrow-road POI detour.

    Returns:
        Shared road features with constant 0.06 seconds/metre travel rates.
    """
    nodes = pd.DataFrame(
        [
            (0, 16.0, 108.0),
            (1, 16.0, 108.003),
            (2, 16.0, 108.006),
            (3, 16.003, 108.003),
        ],
        columns=["node_idx", "lat", "lon"],
    )
    edges = pd.DataFrame(
        [
            (0, 1, "AJ", 60),
            (1, 2, "JB", 60),
            (1, 3, "JP", 60),
            (3, 1, "PJ", 60),
            (1, 0, "JA", 60),
        ],
        columns=["from_idx", "to_idx", "road_name", "length_m"],
    )
    points = [
        {"id": id, "name": id, "lat": lat, "lon": lon, "kind": kind, "score": score}
        for id, lat, lon, kind, score in [
            ("A", 16.0, 108.0, "hub", 0),
            ("B", 16.0, 108.006, "hub", 0),
            ("P", 16.003, 108.003, "poi", 5),
        ]
    ]
    params = {
        "s_per_m": {
            s: {
                g: 0.06
                for g in ["fast", "primary", "secondary", "tertiary", "minor", "local"]
            }
            for s in ["morning", "ban_noon", "afternoon", "ban_evening", "night"]
        }
    }
    graph = RoadGraph(nodes, edges, {"jp": 4, "pj": 4}, points, time_params=params)
    return RoadFeatures(graph)


def choose(env: RoadEnv, node: int) -> int:
    """Find the valid outgoing slot leading to a requested test node.

    Args:
        env: Trip whose current outgoing slots are inspected.
        node: Desired destination node position for one edge.

    Returns:
        First valid action slot reaching node.

    Raises:
        StopIteration: No currently valid outgoing edge reaches node.
    """
    obs = env.observe()
    return next(
        i
        for i, edge in enumerate(obs["edges"])
        if edge >= 0 and env.graph.v[edge] == node and obs["mask"][i]
    )


def test_edge_rewards_unique_pois_narrow_penalty_and_shared_renderer(
    features: RoadFeatures,
) -> None:
    """Verify POIs count once, narrow travel costs reward and replay agrees."""
    config = RoadPPOConfig(max_episode_steps=20)
    env = RoadEnv(features, 0, 2, 480, 10, config)
    env.step(choose(env, 1))
    reward, _ = env.step(choose(env, 3))
    assert reward == pytest.approx(
        5 - (config.time_penalty + config.narrow_penalty) * 3.6 / 60
    )
    env.step(choose(env, 1))
    reward, _ = env.step(choose(env, 3))
    assert reward < 0
    env.step(choose(env, 1))
    reward, done = env.step(choose(env, 2))
    assert done and env.success and reward > 0
    snapshot = features.graph.render_path(["A", "B"], env.path, 480, 600)
    assert snapshot["score"] == 5 and snapshot["duration_min"] == pytest.approx(
        env.elapsed_s / 60
    )
    assert snapshot["collected"] == ["P"]
    with pytest.raises(ValueError):
        env.step(0)


def test_mask_invalid_padding_and_provably_insufficient_time(
    features: RoadFeatures,
) -> None:
    """Verify masked slots have zero action probability during prediction."""
    config = RoadPPOConfig(hidden_dim=8)
    env = RoadEnv(features, 0, 2, 480, 10, config)
    obs = env.observe()
    assert obs["mask"].tolist() == [True, False, False]
    with pytest.raises(ValueError):
        env.step(1)
    model = make_model(features, RoadTrainingEnv(features, config), config)
    batch = features.pack(env)[None]
    with torch.no_grad():
        distribution = model.policy.get_distribution(torch.as_tensor(batch))
        probs = distribution.distribution.probs.numpy()[0]
    assert probs.tolist() == [1.0, 0.0, 0.0]
    assert torch.isfinite(distribution.entropy()).all()
    assert {int(model.predict(features.pack(env))[0]) for _ in range(20)} == {0}
    short = RoadEnv(features, 0, 2, 480, 0.1, config)
    assert not short.observe()["mask"].any()


def test_gym_adapter_terminates_failures_without_timeout_bootstrap(
    features: RoadFeatures,
) -> None:
    """Check Gymnasium seeding, shapes and terminal failure semantics."""
    config = RoadPPOConfig(max_episode_steps=1)
    env = RoadTrainingEnv(features, config, scenario=(0, 2, 480, 10))
    check_env(env)
    obs, _ = env.reset(seed=7)
    assert env.observation_space.contains(obs)
    assert obs.shape == (features.obs_size,)
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and not info["success"] and reward < -19
    with pytest.raises(RuntimeError):
        env.step(0)
    # Reset uses Gym's generator, so identical seeds reproduce a sampled trip.
    sampled = RoadTrainingEnv(features, config)
    first = sampled.reset(seed=21)[0]
    assert np.array_equal(first, sampled.reset(seed=21)[0])


def test_step_limit_is_failure_not_success(features: RoadFeatures) -> None:
    """Ensure exhausting the edge cap is penalized as an unsuccessful trip."""
    env = RoadEnv(features, 0, 2, 480, 10, RoadPPOConfig(max_episode_steps=1))
    reward, done = env.step(0)
    assert done and not env.success and reward < -19


def test_sb3_bootstraps_rollout_cuts_and_zeros_terminal_failures(
    features: RoadFeatures,
) -> None:
    """Check SB3 value targets at genuine terminals and mid-trip rollout cuts."""
    nodes = pd.DataFrame(
        [(i, 16.0, 108.0 + i * 0.001) for i in range(5)],
        columns=["node_idx", "lat", "lon"],
    )
    edges = pd.DataFrame(
        [(i, i + 1, "chain", 60.0) for i in range(4)],
        columns=["from_idx", "to_idx", "road_name", "length_m"],
    )
    chain = RoadFeatures(
        RoadGraph(nodes, edges, {}, [], time_params=features.graph.time_params)
    )

    class CheckBootstrap(BaseCallback):
        def _on_step(self) -> bool:
            """Allow SB3 to collect every interaction in the test rollout.

            Returns:
                True to continue collection.
            """
            return True

        def _on_rollout_end(self) -> None:
            """Check the final value target before the optimizer changes its critic."""
            buffer = self.model.rollout_buffer
            done = bool(self.locals["dones"][0])
            expected = buffer.rewards[-1, 0]
            if not done:
                # Callback locals were captured during the final step. Compute
                # V(next_obs) before optimization rather than reusing V(obs).
                with torch.no_grad():
                    next_value = self.model.policy.predict_values(
                        torch.as_tensor(self.locals["new_obs"])
                    )
                expected += self.model.gamma * float(next_value.item())
            assert buffer.returns[-1, 0] == pytest.approx(expected)
            self.was_terminal = done

    for cap, terminal in [(1, True), (100, False)]:
        config = RoadPPOConfig(
            total_steps=2,
            rollout_steps=2,
            minibatch_size=2,
            epochs=1,
            hidden_dim=8,
            max_episode_steps=cap,
        )
        # Four edges to the goal: a two-edge rollout must end mid-trip, unless
        # the one-edge cap deliberately terminates each attempt as a failure.
        env = RoadTrainingEnv(chain, config, scenario=(0, 4, 480, 10))
        model = make_model(chain, env, config)
        callback = CheckBootstrap()
        model.learn(2, callback=callback)
        assert callback.was_terminal == terminal


def test_training_changes_gnn_weights_and_checkpoint_roundtrip(
    features: RoadFeatures, tmp_path: Path
) -> None:
    """Verify gradients, rollout rounding, reproducibility and stateful resume."""
    config = RoadPPOConfig(
        total_steps=5,  # SB3 completes two four-step rollouts (eight actions).
        rollout_steps=4,
        minibatch_size=2,
        epochs=2,
        hidden_dim=8,
        max_episode_steps=12,
        seed=7,
        threads=1,
    )
    initial = make_model(features, RoadTrainingEnv(features, config), config)
    weights = initial.policy.encoder.layer_0.weight.detach().clone()
    path = tmp_path / "road.zip"
    model, history = train(features, config, path)
    assert len(history) == 2 and history[-1]["steps"] == 8
    assert not torch.equal(weights, model.policy.encoder.layer_0.weight)
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
    loaded, saved_config = load_policy(path, features)
    assert saved_config == config
    obs = torch.as_tensor(features.pack(RoadEnv(features, 0, 2, 480, 10, config))[None])
    with torch.no_grad():
        first = model.policy.get_distribution(obs).distribution.logits
        second = loaded.policy.get_distribution(obs).distribution.logits
    assert torch.equal(first, second)
    optimizer = model.policy.optimizer.state_dict()["state"]
    restored = loaded.policy.optimizer.state_dict()["state"]
    for key in optimizer:
        for name in optimizer[key]:
            assert torch.equal(optimizer[key][name], restored[key][name])
    assert read_checkpoint(path, features)["updates"] == 2
    again, _ = train(features, config, tmp_path / "again.zip")
    assert all(
        torch.equal(p, q)
        for p, q in zip(model.policy.parameters(), again.policy.parameters())
    )
    _, resumed = train(features, config, path, resume=path)
    assert resumed[-1]["steps"] == 16 and resumed[-1]["updates"] == 4
    old = fingerprint(features.graph)
    features.graph.points["A"]["routing_available"] = True
    features.graph.points["A"]["snap_distance_m"] = 0.0
    assert fingerprint(features.graph) == old
    features.graph.times["morning"][0] += 1
    with pytest.raises(ValueError, match="does not match"):
        load_policy(path, features)


def test_masked_actor_encoder_receives_gradients(features: RoadFeatures) -> None:
    """Ensure actor gradients reach its trainable full-graph encoder."""
    config = RoadPPOConfig(hidden_dim=8)
    model = make_model(features, RoadTrainingEnv(features, config), config)
    obs = torch.as_tensor(features.pack(RoadEnv(features, 1, 2, 480, 10, config))[None])
    model.policy.set_training_mode(True)
    log_prob = model.policy.get_distribution(obs).log_prob(torch.tensor([0]))
    log_prob.sum().backward()
    assert any(
        p.grad is not None and bool(torch.any(p.grad != 0))
        for p in model.policy.encoder.parameters()
    )


def test_inference_has_no_dijkstra_fallback_and_renders_actual_edges(
    features: RoadFeatures,
) -> None:
    """Verify deterministic policy geometry and explicit unsuccessful inference."""
    config = RoadPPOConfig(hidden_dim=8, max_episode_steps=20)
    model = make_model(features, RoadTrainingEnv(features, config), config)
    with torch.no_grad():
        for parameter in model.policy.parameters():
            parameter.zero_()
    model.policy.clear_embedding_cache()
    # Equal logits take the first available edge: A->junction->B.
    result = infer(features, model, config, "A", "B", 480, 10)
    assert result["duration_min"] == pytest.approx(0.12)
    assert result["score"] == 0
    report = evaluate(features, model, config, [(0, 2, 480, 10)])
    assert report["completed"] == report["count"] == 1
    with pytest.raises(ValueError, match="did not reach"):
        infer(features, model, RoadPPOConfig(max_episode_steps=1), "A", "B", 480, 10)


def test_edge_clock_changes_within_a_connection(features: RoadFeatures) -> None:
    """Compare heuristic/PPO edge clocks across a traffic-slot boundary."""
    graph = features.graph
    graph.times["morning"][:] = 90.0
    graph.times["ban_noon"][:] = 180.0
    problem = RoadProblem(graph, "A", "B", 5, start_minute=629, all_pois=False)
    assert problem.direct.duration_s == 270
    result = graph.snapshot(problem, problem.direct)
    assert result["duration_min"] == 4.5
    env = RoadEnv(features, 0, 2, 629, 5, RoadPPOConfig())
    env.step(choose(env, 1))
    env.step(choose(env, 2))
    assert env.success and env.elapsed_s == 270
    with pytest.raises(ValueError, match="time budget"):
        RoadProblem(graph, "A", "B", 4, start_minute=629, all_pois=False)


def test_all_outgoing_edges_are_kept_not_truncated() -> None:
    """Retain every outgoing choice even when a node has ten edges."""
    nodes = pd.DataFrame(
        [(i, 16.0, 108.0 + i * 0.001) for i in range(11)],
        columns=["node_idx", "lat", "lon"],
    )
    edges = pd.DataFrame(
        [(0, i, "road", 100.0) for i in range(1, 11)],
        columns=["from_idx", "to_idx", "road_name", "length_m"],
    )
    points = [
        {"id": "A", "name": "A", "kind": "hub", "lat": 16.0, "lon": 108.0, "score": 0}
    ]
    graph = RoadGraph(nodes, edges, {}, points)
    features = RoadFeatures(graph)
    obs = RoadEnv(features, 0, 10, 480, 10, RoadPPOConfig()).observe()
    assert features.k == 10 and obs["mask"][9]


def test_curriculum_excludes_heldout_start_goal_pairs(features: RoadFeatures) -> None:
    """Exclude ordered held-out endpoint pairs throughout the curriculum."""
    config = RoadPPOConfig(curriculum_steps=100)
    rng = np.random.default_rng(15)
    heldout = [(0, 2, 480, 10)]
    for steps in (0, 50, 100):
        for _ in range(10):
            env = sample_env(features, rng, config, steps, heldout)
            assert (env.source, env.target) != (0, 2)
            assert env.observe()["mask"].any()


def test_cached_edge_clock_matches_direct_replay_across_days(
    features: RoadFeatures,
) -> None:
    """Compare cached leg timing with per-edge replay through multiple days."""
    from busroute.routing.road_graph import RoadLeg
    from busroute.utils.timeslots import slot_of

    rng = np.random.default_rng(31)
    graph = features.graph
    for slot in graph.times:
        graph.times[slot][:] = rng.uniform(1, 90000, len(graph.edges))
    leg = RoadLeg(0, 0, 0, tuple(rng.integers(len(graph.edges), size=30)), 0)
    for minute in (0, 299, 300, 629, 630, 719, 720, 989, 990, 1139, 1140, 1439):
        elapsed = float(rng.uniform(0, 86400))
        expected = elapsed
        for edge in leg.edges:
            expected += graph.times[slot_of((minute + int(expected // 60)) % 1440)][
                edge
            ]
        assert graph.advance_leg(leg, minute, elapsed) == pytest.approx(expected)


def test_encoder_cache_is_rollout_only_and_invalidated_after_training(
    features: RoadFeatures,
) -> None:
    """Cache static embeddings in evaluation, but rebuild them for each gradient pass."""
    config = RoadPPOConfig(hidden_dim=8, threads=1)
    model = make_model(features, RoadTrainingEnv(features, config), config)
    obs = torch.as_tensor(features.pack(RoadEnv(features, 1, 2, 480, 10, config))[None])
    model.policy.set_training_mode(False)
    with patch.object(
        model.policy.encoder, "forward", wraps=model.policy.encoder.forward
    ) as encoder:
        with torch.no_grad():
            model.policy.get_distribution(obs)
            model.policy.predict_values(obs)
            model.policy.get_distribution(obs)
        assert encoder.call_count == 1
        model.policy.set_training_mode(True)
        for _ in range(2):
            model.policy.evaluate_actions(obs, torch.tensor([0]))
        assert encoder.call_count == 3
        model.policy.set_training_mode(False)
        with torch.no_grad():
            model.policy.get_distribution(obs)
        assert encoder.call_count == 4


def test_sbx_checkpoint_is_rejected_before_model_deserialization(
    features: RoadFeatures, tmp_path: Path
) -> None:
    """Give a clear retraining error for the previous backend's native ZIPs."""
    import json
    from zipfile import ZipFile

    path = tmp_path / "old.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "data",
            json.dumps(
                {
                    "road_metadata": {
                        "version": 2,
                        "graph_sha256": fingerprint(features.graph),
                    }
                }
            ),
        )
    with pytest.raises(ValueError, match="SBX"):
        load_policy(path, features)
