"""Exercise OD routing rewards, action masks, PPO training and solver interfaces.

The small hand-built graph covers edge cases; the fully connected CSV in the
integration test lets a tiny PPO rollout reach a destination from any hub.
Times passed to solvers are minutes after midnight and budgets are in seconds.
"""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from busroute.routing.od_env import ODRouteEnv
from busroute.routing.od_evaluation import evaluate, greedy_route, shortest_route
from busroute.routing.orienteering import Leg, ODMatrix


def small_matrix():
    """Two hubs and two POIs; travelling to P1 also passes through P2."""
    legs = {}
    for slot in ("morning", "ban_noon", "afternoon", "ban_evening", "night"):
        legs[("H1", "P1", slot)] = Leg(60, frozenset({"P2"}))
        legs[("P1", "H2", slot)] = Leg(60, frozenset({"P2"}))
        legs[("H1", "H2", slot)] = Leg(180, frozenset())
        legs[("P2", "H2", slot)] = Leg(60, frozenset())
    return ODMatrix(legs, {"P1": 4.0, "P2": 3.0}, {"H1", "H2", "P1", "P2"})


def test_env_passed_pois_mask_and_arrival():
    """Passing a POI collects it once, including across a time-slot boundary."""
    env = ODRouteEnv(small_matrix(), "H1", "H2", 629, 120)
    assert env.action_mask()["P1"]
    obs, reward, done, _ = env.step("P1")
    assert obs["slot"] == "ban_noon"
    assert obs["collected"] == {"P1", "P2"}
    assert not obs["mask"]["P2"]
    assert reward == pytest.approx(6.99)  # P1 + P2 scores minus travel penalty.
    obs, reward, done, info = env.step("H2")
    assert done and info["success"] and reward == pytest.approx(9.99)  # Arrival bonus.
    assert env.matrix.evaluate(env.route, 629, 120).score == 7
    with pytest.raises(ValueError):
        env.step("H2")


def test_no_action_and_failure_report():
    """An insufficient budget leaves no action and is reported as a failure."""
    matrix = small_matrix()
    env = ODRouteEnv(matrix, "H1", "H2", 480, 30)
    assert not any(env.action_mask().values())
    report = evaluate(matrix, [("H1", "H2", 480, 30)], {"shortest": shortest_route})
    assert report["shortest"]["completed"] == 0
    assert report["shortest"]["results"][0]["error"]


def test_slot_change_can_cause_dead_end_and_is_penalized():
    """A feasible estimate can become a dead end after a passed POI is collected."""
    legs = {
        ("H1", "P1", "morning"): Leg(60, frozenset()),
        ("P1", "P2", "ban_noon"): Leg(60, frozenset()),
        ("P2", "H2", "ban_noon"): Leg(60, frozenset()),
    }
    matrix = ODMatrix(legs, {"P1": 4.0, "P2": 3.0}, {"H1", "H2", "P1", "P2"})
    env = ODRouteEnv(matrix, "H1", "H2", 629, 180)
    assert env.action_mask()["P1"]
    # The return estimate includes P2, but P2 is collected while reaching P1.
    legs["H1", "P1", "morning"] = Leg(60, frozenset({"P2"}))
    _, reward, done, info = env.step("P1")
    assert done and not info["success"]
    assert reward == pytest.approx(4 + 3 - 0.01 - 20)


def test_greedy_and_shortest_use_shared_evaluator():
    """Baseline results use the same scoring and feasibility rules as the matrix."""
    matrix = small_matrix()
    for solver in (greedy_route, shortest_route):
        result = solver(matrix, "H1", "H2", 480, 180)
        assert result == matrix.evaluate(result.route, 480, 180)


def test_sbx_training_masks_checkpoint_and_inference(tmp_path):
    """Train a tiny policy, then check masking, checkpoint integrity and solver APIs.

    Four steps are enough to exercise a PPO update, not to measure route quality.
    The dependency is optional so non-ML routing tests can still run without SBX.
    """
    pytest.importorskip("sbx")
    import jax.numpy as jnp
    import numpy as np
    from stable_baselines3.common.env_checker import check_env
    from busroute.routing.ppo_od import (
        PPOConfig,
        Features,
        ODTrainingEnv,
        PPOSearch,
        OnlinePPOSearch,
        infer,
        load_coordinates,
        load_policy,
        train,
    )
    from busroute.routing.metaheuristics import _Search

    # All directed legs exist in every time slot, with consistent coordinates.
    path = tmp_path / "matrix.csv"
    fields = [
        "from_id",
        "to_id",
        "from_lat",
        "from_lon",
        "to_lat",
        "to_lon",
        "to_type",
        "to_poi_score",
        "slot",
        "feasible",
        "duration_s",
        "pois_passed",
    ]
    points = ["H1", "H2", "P1", "P2"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i, u in enumerate(points):
            for j, v in enumerate(points):
                if u == v:
                    continue
                for slot in (
                    "morning",
                    "ban_noon",
                    "afternoon",
                    "ban_evening",
                    "night",
                ):
                    writer.writerow(
                        dict(
                            from_id=u,
                            to_id=v,
                            from_lat=i,
                            from_lon=i,
                            to_lat=j,
                            to_lon=j,
                            to_type="poi" if v.startswith("P") else "hub",
                            to_poi_score=4 if v.startswith("P") else 0,
                            slot=slot,
                            feasible=1,
                            duration_s=60,
                            pois_passed="",
                        )
                    )
    matrix = ODMatrix.from_csv(path)
    checkpoint = tmp_path / "policy.zip"
    cfg = PPOConfig(
        total_steps=4, rollout_steps=4, epochs=1, minibatch_size=2, hidden_dim=8
    )
    # Verify a real optimizer update survives saving and loading.
    history = train(matrix, str(path), cfg, str(checkpoint), eval_count=2)
    model = load_policy(str(checkpoint), str(path))
    assert history[-1]["steps"] == 4
    assert model._n_updates == 1
    features = Features(matrix, load_coordinates(path))
    # Gym resets are reproducible and invalid actions terminate with a penalty.
    check_env(ODTrainingEnv(features), warn=True)
    gym_env = ODTrainingEnv(features)
    first, first_info = gym_env.reset(seed=42)
    second, second_info = gym_env.reset(seed=42)
    assert first_info == second_info
    np.testing.assert_array_equal(first, second)
    invalid = features.index[gym_env.route_env.current]
    _, penalty, terminated, truncated, info = gym_env.step(invalid)
    assert terminated and not truncated and info["invalid_action"] and penalty == -20
    # Inspect the actor distribution as well as sampled actions: masked actions
    # must have zero probability, while entropy must remain finite.
    env = ODRouteEnv(matrix, "H1", "H2", 480, 3600)
    observation = features.make(env)
    dist = model.policy.actor_state.apply_fn(
        model.policy.actor_state.params, jnp.array(observation[None, :])
    )
    probs = np.asarray(dist.probs_parameter())[0]
    mask = np.array(list(env.action_mask().values()))
    assert np.all(probs[~mask] == 0)
    assert np.isfinite(np.asarray(dist.entropy())).all()
    for _ in range(20):
        action, _ = model.predict(observation, deterministic=False)
        assert mask[int(action)]
    # PPO and the other solvers return routes evaluated by the same rules.
    result = infer(model, features, "H1", "H2", 480, 3600)
    assert result == matrix.evaluate(result.route, 480, 3600)
    solver = PPOSearch(model, features)
    assert isinstance(solver, _Search)
    assert solver(matrix, "H1", "H2", 480, 3600) == result
    with pytest.raises(ValueError, match="supplied matrix"):
        solver(small_matrix(), "H1", "H2", 480, 3600)
    with pytest.raises(ValueError, match="feasible route"):
        solver(matrix, "H1", "H2", 480, 1)
    # Each request collects fresh experience and updates the same in-memory policy.
    online = OnlinePPOSearch(features, cfg)
    live_result = online(matrix, "H1", "H2", 480, 3600)
    assert live_result == matrix.evaluate(live_result.route, 480, 3600)
    live_model = online.model
    assert live_model.num_timesteps == 4 and live_model._n_updates == 1
    online(matrix, "H2", "H1", 480, 3600)
    assert online.model is live_model
    assert live_model.num_timesteps == 8 and live_model._n_updates == 2
    with pytest.raises(ValueError, match="No valid initial action"):
        online(matrix, "H1", "H2", 480, 1)
    root = Path(__file__).resolve().parents[1]
    online_run = subprocess.run(
        [sys.executable, str(root / "scripts/online_ppo.py"), "--matrix", str(path),
         "--start", "H1", "--end", "H2", "--budget-min", "60", "--steps", "8",
         "--rollout-steps", "4", "--minibatch-size", "2", "--epochs", "1",
         "--hidden-dim", "8"],
        text=True, capture_output=True, check=True,
    )
    live_json = json.loads(online_run.stdout)
    assert live_json == matrix.evaluate(live_json["route"], 480, 3600).to_dict()
    logs = [json.loads(line) for line in online_run.stderr.splitlines()]
    assert [entry["event"] for entry in logs] == ["training_start", "rollout", "rollout", "training_end"]
    assert [entry["steps"] for entry in logs] == [0, 4, 8, 8]
    assert [entry["completed_updates"] for entry in logs[1:]] == [0, 1, 2]
    assert all(entry["episodes"] > 0 and entry["mean_episode_reward"] is not None for entry in logs[1:3])
    ppo_output = subprocess.check_output(
        [
            sys.executable,
            str(root / "scripts/ppo_routes.py"),
            "--matrix",
            str(path),
            "infer",
            "--checkpoint",
            str(checkpoint),
            "--start",
            "H1",
            "--end",
            "H2",
            "--budget-min",
            "60",
        ],
        text=True,
    )
    ppo_json = json.loads(ppo_output)
    assert ppo_json == result.to_dict()
    for algorithm in ("alns", "vns", "ppo"):
        extra = ["--checkpoint", str(checkpoint)] if algorithm == "ppo" else []
        output = subprocess.check_output(
            [
                sys.executable,
                str(root / "scripts/solve_orienteering.py"),
                "--matrix",
                str(path),
                "--algorithm",
                algorithm,
                "--start",
                "H1",
                "--end",
                "H2",
                "--budget-min",
                "60",
                "--iterations",
                "0",
            ] + extra,
            text=True,
        )
        actual = json.loads(output)
        assert actual.keys() == ppo_json.keys()
        assert actual == matrix.evaluate(actual["route"], 480, 3600).to_dict()
        if algorithm == "ppo":
            assert actual == ppo_json
    # Even a byte-level change to the matrix invalidates the checkpoint.
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="matrix hash"):
        load_policy(str(checkpoint), str(path))
