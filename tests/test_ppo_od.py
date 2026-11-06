import csv

import pytest

from busroute.routing.od_env import ODRouteEnv
from busroute.routing.od_evaluation import evaluate, greedy_route, shortest_route
from busroute.routing.orienteering import Leg, ODMatrix


def small_matrix():
    legs = {}
    for slot in ("morning", "ban_noon", "afternoon", "ban_evening", "night"):
        legs[("H1", "P1", slot)] = Leg(60, frozenset({"P2"}))
        legs[("P1", "H2", slot)] = Leg(60, frozenset({"P2"}))
        legs[("H1", "H2", slot)] = Leg(180, frozenset())
        legs[("P2", "H2", slot)] = Leg(60, frozenset())
    return ODMatrix(legs, {"P1": 4., "P2": 3.}, {"H1", "H2", "P1", "P2"})


def test_env_passed_pois_mask_and_arrival():
    env = ODRouteEnv(small_matrix(), "H1", "H2", 629, 120)
    assert env.action_mask()["P1"]
    obs, reward, done, _ = env.step("P1")
    assert obs["slot"] == "ban_noon"
    assert obs["collected"] == {"P1", "P2"}
    assert not obs["mask"]["P2"]
    assert reward == pytest.approx(6.99)
    obs, reward, done, info = env.step("H2")
    assert done and info["success"] and reward == pytest.approx(9.99)
    assert env.matrix.evaluate(env.route, 629, 120).score == 7
    with pytest.raises(ValueError):
        env.step("H2")


def test_no_action_and_failure_report():
    matrix = small_matrix()
    env = ODRouteEnv(matrix, "H1", "H2", 480, 30)
    assert not any(env.action_mask().values())
    report = evaluate(matrix, [("H1", "H2", 480, 30)], {"shortest": shortest_route})
    assert report["shortest"]["completed"] == 0
    assert report["shortest"]["results"][0]["error"]


def test_slot_change_can_cause_dead_end_and_is_penalized():
    legs = {
        ("H1", "P1", "morning"): Leg(60, frozenset()),
        ("P1", "P2", "ban_noon"): Leg(60, frozenset()),
        ("P2", "H2", "ban_noon"): Leg(60, frozenset()),
    }
    matrix = ODMatrix(legs, {"P1": 4., "P2": 3.}, {"H1", "H2", "P1", "P2"})
    env = ODRouteEnv(matrix, "H1", "H2", 629, 180)
    assert env.action_mask()["P1"]
    # The return estimate includes P2, but P2 is collected while reaching P1.
    legs["H1", "P1", "morning"] = Leg(60, frozenset({"P2"}))
    _, reward, done, info = env.step("P1")
    assert done and not info["success"]
    assert reward == pytest.approx(4 + 3 - .01 - 20)


def test_greedy_and_shortest_use_shared_evaluator():
    matrix = small_matrix()
    for solver in (greedy_route, shortest_route):
        result = solver(matrix, "H1", "H2", 480, 180)
        assert result == matrix.evaluate(result.route, 480, 180)


def test_sbx_training_masks_checkpoint_and_inference(tmp_path):
    pytest.importorskip("sbx")
    import jax.numpy as jnp
    import numpy as np
    from stable_baselines3.common.env_checker import check_env
    from busroute.routing.ppo_od import (PPOConfig, Features, ODTrainingEnv, infer,
                                          load_coordinates, load_policy, train)
    path = tmp_path / "matrix.csv"
    fields = ["from_id", "to_id", "from_lat", "from_lon", "to_lat", "to_lon", "to_type", "to_poi_score", "slot", "feasible", "duration_s", "pois_passed"]
    points = ["H1", "H2", "P1", "P2"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for i, u in enumerate(points):
            for j, v in enumerate(points):
                if u == v:
                    continue
                for slot in ("morning", "ban_noon", "afternoon", "ban_evening", "night"):
                    writer.writerow(dict(from_id=u, to_id=v, from_lat=i, from_lon=i, to_lat=j, to_lon=j,
                                         to_type="poi" if v.startswith("P") else "hub", to_poi_score=4 if v.startswith("P") else 0,
                                         slot=slot, feasible=1, duration_s=60, pois_passed=""))
    matrix = ODMatrix.from_csv(path)
    checkpoint = tmp_path / "policy.zip"
    cfg = PPOConfig(total_steps=4, rollout_steps=4, epochs=1, minibatch_size=2, hidden_dim=8)
    history = train(matrix, str(path), cfg, str(checkpoint), eval_count=2)
    model = load_policy(str(checkpoint), str(path))
    assert history[-1]["steps"] == 4
    assert model._n_updates == 1
    features = Features(matrix, load_coordinates(path))
    check_env(ODTrainingEnv(features), warn=True)
    gym_env = ODTrainingEnv(features)
    first, first_info = gym_env.reset(seed=42)
    second, second_info = gym_env.reset(seed=42)
    assert first_info == second_info
    np.testing.assert_array_equal(first, second)
    invalid = features.index[gym_env.route_env.current]
    _, penalty, terminated, truncated, info = gym_env.step(invalid)
    assert terminated and not truncated and info["invalid_action"] and penalty == -20
    env = ODRouteEnv(matrix, "H1", "H2", 480, 3600)
    observation = features.make(env)
    dist = model.policy.actor_state.apply_fn(model.policy.actor_state.params, jnp.array(observation[None, :]))
    probs = np.asarray(dist.probs_parameter())[0]
    mask = np.array(list(env.action_mask().values()))
    assert np.all(probs[~mask] == 0)
    assert np.isfinite(np.asarray(dist.entropy())).all()
    for _ in range(20):
        action, _ = model.predict(observation, deterministic=False)
        assert mask[int(action)]
    result = infer(model, features, "H1", "H2", 480, 3600)
    assert result == matrix.evaluate(result.route, 480, 3600)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="matrix hash"):
        load_policy(str(checkpoint), str(path))
