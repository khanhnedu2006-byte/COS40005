"""SB3 PPO training, checkpoint validation and road-route inference.

Read ``road_ppo_env`` for routing rules and observation construction, and
``road_ppo_policy`` for the Torch graph networks. This module deliberately has
no handwritten PPO loss, GAE calculation or optimizer loop: ``stable_baselines3.PPO.learn``
owns them. The public helpers are shared by the CLI and web worker.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypedDict, cast
from zipfile import BadZipFile, ZipFile

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from typing_extensions import NotRequired

from busroute.routing.road_graph import RoadGraph
from busroute.routing.road_ppo_env import (
    RoadEnv,
    RoadFeatures,
    RoadPPOConfig,
    RoadScenario,
    RoadTrainingEnv,
    sample_env,
)
from busroute.routing.road_ppo_policy import RoadPolicy, StaticRoadGraph


class TrainingRecord(TypedDict):
    """Metrics for one rollout, completed with SB3 optimizer statistics.

    Attributes:
        steps: Cumulative edge interactions, including earlier sessions.
        mean_reward: Mean shaped reward per interaction in this rollout.
        episodes: Trips ending in this rollout, successful or otherwise.
        successes: Trips reaching their goal within budget in this rollout.
        updates: Completed rollout updates; absent while an update is pending.
        policy_gradient_loss: SB3's policy loss after the update, if logged.
        value_loss: SB3's value loss, if logged.
        entropy_loss: SB3's entropy loss, if logged.
        approx_kl: SB3's approximate policy KL divergence, if logged.
    """

    steps: int
    mean_reward: float
    episodes: int
    successes: int
    updates: NotRequired[int]
    policy_gradient_loss: NotRequired[float]
    value_loss: NotRequired[float]
    entropy_loss: NotRequired[float]
    approx_kl: NotRequired[float]


class ScenarioResult(TypedDict):
    """One evaluation trip, with a nullable score for unfinished routes.

    Attributes:
        source: Start node position in the loaded graph.
        target: Goal node position in the loaded graph.
        departure_minute: Local minute after midnight.
        budget_min: Allowed travel time in minutes.
        success: Whether the policy reached the goal within its limits.
        score: Unique POI score on success; None on failure.
        duration_min: Travel time of the attempted path, in minutes.
        steps: Edges traversed, including on failed trips.
        distance_km: Physical length of the attempted path, in kilometres.
        narrow_m: Attempted path length on known roads narrower than 5.5 m.
        unknown_width_m: Attempted path length without known width, in metres.
    """

    source: int
    target: int
    departure_minute: int
    budget_min: float
    success: bool
    score: float | None
    duration_min: float
    steps: int
    distance_km: float
    narrow_m: float
    unknown_width_m: float


class TrainingSummary(TypedDict):
    """JSON training provenance accompanying an evaluation report.

    Attributes:
        history: Metrics from this training invocation's completed updates.
        config: Effective serialized hyperparameters and reward coefficients.
        checkpoint: Saved SB3 ZIP path.
        nodes: Number of graph nodes retained for training.
        edges: Number of directed graph edges retained for training.
        backend: Algorithm implementation identifier, currently 'sb3'.
        device: Torch device used for learning, currently 'cpu'.
        requested_steps: Additional interaction budget before rollout rounding.
    """

    history: list[TrainingRecord]
    config: dict[str, int | float]
    checkpoint: str
    nodes: int
    edges: int
    backend: str
    device: str
    requested_steps: int


class EvaluationReport(TypedDict):
    """Evaluation results with optional training provenance.

    Attributes:
        scenarios: Per-trip metrics in input order.
        completed: Count of successful trips.
        count: Number of attempted trips.
        training: Training summary; present only for training CLI reports.
    """

    scenarios: list[ScenarioResult]
    completed: int
    count: int
    training: NotRequired[TrainingSummary]


ProgressCallback: TypeAlias = Callable[[TrainingRecord], None]


class _RoadModelState(Protocol):
    """Extra attributes serialized by SB3 but absent from its upstream hints."""

    road_metadata: dict[str, Any]


if TYPE_CHECKING:
    from Web.types import RouteSnapshot


# Keep the CLI/web import surface stable while the implementation is split by
# responsibility. The environment and policy modules can also be read directly.
__all__ = [
    "RoadEnv",
    "RoadFeatures",
    "RoadPPOConfig",
    "RoadTrainingEnv",
    "checkpoint_path",
    "evaluate",
    "fingerprint",
    "infer",
    "infer_nodes",
    "load_policy",
    "make_model",
    "read_checkpoint",
    "sample_env",
    "train",
]


def fingerprint(graph: RoadGraph) -> str:
    """Hash the graph ordering, POIs, widths and travel-time semantics.

    Args:
        graph: Full road graph whose attributes bind a trained checkpoint.

    Returns:
        SHA-256 hex digest. Web-only snapping/availability annotations are
        excluded so CLI checkpoints remain usable by the web worker.
    """
    digest = hashlib.sha256(b"road-ppo-edge-clock-v1")
    for array in (
        graph.nodes.node_idx.to_numpy(),
        graph.coords,
        graph.u,
        graph.v,
        graph.lengths,
        graph.widths,
        graph.edges.edge_idx.to_numpy(),
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(json.dumps(graph.time_params, sort_keys=True).encode())
    for slot in sorted(graph.times):
        digest.update(np.ascontiguousarray(graph.times[slot]).tobytes())
    digest.update(",".join(map(str, graph.masks)).encode())
    # Web-only availability/snap annotations must not invalidate CLI checkpoints.
    fields = ("id", "name", "kind", "lat", "lon", "score")
    canonical = {
        key: {field: point[field] for field in fields}
        for key, point in graph.points.items()
    }
    digest.update(json.dumps(canonical, sort_keys=True).encode())
    return digest.hexdigest()


def checkpoint_path(path: str | Path) -> Path:
    """Normalize an SB3 ZIP path and reject incompatible file suffixes.

    Args:
        path: Checkpoint filename, optionally without the .zip suffix.

    Returns:
        Path with a .zip suffix, without reading or creating the file.

    Raises:
        ValueError: A supplied suffix denotes a non-SB3 format such as .pt.
    """
    path = Path(path)
    if path.suffix and path.suffix != ".zip":
        raise ValueError(
            "Road PPO now uses SB3 .zip checkpoints; old .pt weights need retraining"
        )
    return path if path.suffix else path.with_suffix(".zip")


def read_checkpoint(path: str | Path, features: RoadFeatures) -> dict[str, Any]:
    """Read and validate plain JSON metadata before loading the SB3 policy.

    Args:
        path: SB3 road-checkpoint path, optionally without the .zip suffix.
        features: Current graph features used to verify the fingerprint.

    Returns:
        Checkpoint metadata including configuration, counters, held-out cases
        and RNG state. Values crossing the SB3/JSON serialization boundary are
        heterogeneous; NumPy controls the nested bit-generator state schema.

    Raises:
        OSError: The checkpoint cannot be opened.
        ValueError: The archive/metadata, format version or graph hash is invalid.
    """
    try:
        with ZipFile(checkpoint_path(path)) as archive:
            metadata = json.loads(archive.read("data")).get("road_metadata", {})
    except (BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Not a road SB3 checkpoint; retrain legacy .pt models"
        ) from exc
    if metadata.get("version") != 3 or metadata.get("backend") != "sb3":
        raise ValueError(
            "Expected an SB3 road checkpoint; SBX/legacy weights require retraining"
        )
    if metadata.get("graph_sha256") != fingerprint(features.graph):
        raise ValueError(
            "PPO checkpoint does not match this graph, POIs, widths or time model"
        )
    return metadata


def make_model(
    features: RoadFeatures, env: RoadTrainingEnv, config: RoadPPOConfig
) -> PPO:
    """Configure standard SB3 PPO with the masked road actor and critic.

    Args:
        features: Full-graph features, outgoing degree and static node inputs.
        env: Gymnasium road adapter from which SB3 collects training rollouts.
        config: Validated network, optimizer and PPO hyperparameters.

    Returns:
        Initialized SB3 PPO model with one shared trainable graph encoder for
        actor and critic. No training interactions have been collected yet.

    Raises:
        ValueError: The PPO configuration fails validation.
    """
    config.validate()
    torch.set_num_threads(config.threads)
    return PPO(
        RoadPolicy,
        env,
        learning_rate=config.lr,
        n_steps=config.rollout_steps,
        batch_size=config.minibatch_size,
        n_epochs=config.epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_eps,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        seed=config.seed,
        verbose=0,
        device="cpu",
        policy_kwargs={
            "graph": StaticRoadGraph.from_features(features),
            "k": features.k,
            "hidden_dim": config.hidden_dim,
        },
    )


class RoadTrainingCallback(BaseCallback):
    """Save completed optimizer updates and report compact rollout metrics.

    SB3 calls on_rollout_end before optimization. This callback collects
    metrics there, then saves at the next rollout start or training end so
    checkpoints include the completed update's weights and optimizer state.

    Attributes:
        path: Destination SB3 ZIP checkpoint, replaced atomically on each save.
        road_env: Adapter whose sampling RNG is saved for continuation.
        features: Graph features used to fingerprint each checkpoint.
        config: Training settings recorded alongside the model.
        heldout: Fixed evaluation cases excluded from training sampling.
        report: Optional consumer of each completed update's progress record.
        history: Completed progress records for this learning session.
        pending: Rollout statistics awaiting optimization, or None.
        rewards: Per-interaction rewards for the rollout being collected.
        episodes: Trips ending in the current rollout.
        successes: Trips successfully ending in the current rollout.
    """

    def __init__(
        self,
        path: str | Path,
        env: RoadTrainingEnv,
        features: RoadFeatures,
        config: RoadPPOConfig,
        heldout: Sequence[Sequence[int | float]],
        callback: ProgressCallback | None = None,
    ) -> None:
        """Prepare checkpointing and metrics for a single learn call.

        Args:
            path: Output checkpoint path, normalized to SB3's ZIP format.
            env: Adapter supplying trip sampling and the sampler RNG state.
            features: Shared graph features used to bind checkpoint compatibility.
            config: Hyperparameters and rewards to serialize.
            heldout: Fixed evaluation scenarios to serialize.
            callback: Optional progress-record consumer, called after a save.

        Raises:
            ValueError: The checkpoint suffix is incompatible with SB3.
        """
        super().__init__()
        self.path, self.road_env, self.features = checkpoint_path(path), env, features
        self.config, self.heldout, self.report = config, heldout, callback
        self.history: list[TrainingRecord] = []
        self.pending: TrainingRecord | None = None
        self.rewards: list[float] = []
        self.episodes, self.successes = 0, 0

    def _on_step(self) -> bool:
        """Collect rewards and terminal outcomes after an SB3 environment step.

        Returns:
            True so SB3 continues collecting the rollout.

        Raises:
            InterruptedError: The graph's cancellation callback requests a stop.
        """
        self.features.graph.check_cancelled()
        self.rewards.extend(map(float, self.locals["rewards"]))
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            self.episodes += int(done)
            self.successes += int(done and info.get("success", False))
        return True

    def _on_rollout_end(self) -> None:
        """Stage completed rollout metrics until SB3 finishes optimization."""
        self.pending = {
            "steps": self.num_timesteps,
            "mean_reward": float(np.mean(self.rewards)),
            "episodes": self.episodes,
            "successes": self.successes,
        }
        self.rewards, self.episodes, self.successes = [], 0, 0

    def _on_rollout_start(self) -> None:
        """Save the previous rollout's update before collecting another one."""
        self._save_completed_update()

    def _on_training_end(self) -> None:
        """Save the final completed update after SB3 exits its learning loop."""
        self._save_completed_update()

    def _save_completed_update(self) -> None:
        """Persist pending statistics and the already-updated SB3 model.

        The checkpoint also records the sampler/global RNG streams, configuration,
        graph fingerprint and held-out cases. The optional consumer receives the
        same record appended to history. With no pending rollout this is a no-op.

        Raises:
            OSError: Creating or atomically replacing the checkpoint fails.
        """
        if self.pending is None:
            return
        updates = getattr(self.model, "road_metadata", {}).get("updates", 0) + 1
        record = cast(TrainingRecord, dict(self.pending, updates=updates))
        # These metrics come from SB3's optimizer, not a duplicate loss formula.
        for name in ("policy_gradient_loss", "value_loss", "entropy_loss", "approx_kl"):
            value = self.logger.name_to_value.get("train/" + name)
            if value is not None:
                record[name] = float(value)
        numpy_rng = np.random.get_state()
        state = cast(_RoadModelState, self.model)
        state.road_metadata = {
            "version": 3,
            "backend": "sb3",
            "graph_sha256": fingerprint(self.features.graph),
            "config": asdict(self.config),
            "heldout": [list(case) for case in self.heldout],
            "steps": self.num_timesteps,
            "updates": updates,
            "sampler_rng": self.road_env.np_random.bit_generator.state,
            "torch_rng": torch.get_rng_state().tolist(),
            "numpy_rng": [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
            "python_rng": random.getstate(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.stem + ".tmp.zip")
        self.model.save(temporary)
        temporary.replace(self.path)
        self.history.append(record)
        self.pending = None
        if self.report:
            self.report(record)


def load_policy(
    path: str | Path,
    features: RoadFeatures,
    *,
    env: RoadTrainingEnv | None = None,
) -> tuple[PPO, RoadPPOConfig]:
    """Load an SB3 road policy and optimizer state after graph validation.

    Args:
        path: SB3 road-checkpoint ZIP path.
        features: Current graph features used for compatibility checks.
        env: Optional adapter to attach for resumed learning; None is valid
            for inference and evaluation.

    Returns:
        Loaded SB3 PPO model and its validated RoadPPOConfig. Restores Torch, NumPy
        and Python RNG state after SB3 initializes the loaded model.

    Raises:
        OSError: The checkpoint cannot be opened.
        ValueError: The graph, metadata, configuration or model is incompatible.
    """
    metadata = read_checkpoint(path, features)
    config = RoadPPOConfig(**metadata["config"])
    config.validate()
    torch.set_num_threads(config.threads)
    model = PPO.load(checkpoint_path(path), env=env, device="cpu")
    # SB3 reconstructs/initializes the network on load. Restore RNG streams
    # afterwards so continuation keeps policy sampling and minibatch shuffling.
    torch.set_rng_state(torch.tensor(metadata["torch_rng"], dtype=torch.uint8))
    name, values, position, has_gauss, cached_gaussian = metadata["numpy_rng"]
    np.random.set_state(
        (
            name,
            np.asarray(values, dtype=np.uint32),
            position,
            has_gauss,
            cached_gaussian,
        )
    )
    version, values, gaussian = metadata["python_rng"]
    random.setstate((version, tuple(values), gaussian))
    return model, config


def train(
    features: RoadFeatures,
    config: RoadPPOConfig,
    checkpoint: str | Path,
    *,
    resume: str | Path | None = None,
    heldout: Sequence[Sequence[int | float]] = (),
    callback: ProgressCallback | None = None,
) -> tuple[PPO, list[TrainingRecord]]:
    """Train standard SB3 PPO or continue a compatible road checkpoint.

    Args:
        features: Shared full-graph features and cached return-time bounds.
        config: Network/PPO settings and additional requested interactions.
        checkpoint: Output SB3 ZIP path, saved after each completed update.
        resume: Optional input checkpoint; optimizer/RNG/counters are restored
            and sampling starts a new trip rather than restoring an active one.
        heldout: Ordered endpoint cases excluded from fresh training. Resume
            uses the checkpoint's saved cases instead.
        callback: Optional consumer of completed-update progress records.

    Returns:
        Trained SB3 model and progress records for this session. SB3 completes
        full rollouts, so actual interactions may exceed config.total_steps.
        Only total_steps may differ from a resumed checkpoint's configuration.

    Raises:
        ValueError: Settings, checkpoint compatibility or trip sampling fail.
        OSError: Reading or writing a checkpoint fails.
        InterruptedError: The graph's cancellation callback requests a stop.
    """
    config.validate()
    checkpoint_path(checkpoint)  # Fail before loading/building the graph policy.
    metadata = read_checkpoint(resume, features) if resume else {}
    if resume:
        old = dict(metadata["config"], total_steps=config.total_steps)
        if old != asdict(config):
            raise ValueError(
                "Resume settings differ from the checkpoint; keep all settings except total_steps"
            )
        heldout = metadata["heldout"]
    env = RoadTrainingEnv(
        features,
        config,
        heldout,
        steps=metadata.get("steps", 0),
        rng_state=metadata.get("sampler_rng"),
    )
    model = (
        load_policy(resume, features, env=env)[0]
        if resume
        else make_model(features, env, config)
    )
    progress = RoadTrainingCallback(
        checkpoint, env, features, config, heldout, callback
    )
    try:
        model.learn(
            total_timesteps=config.total_steps,
            callback=progress,
            reset_num_timesteps=not resume,
        )
    finally:
        env.close()
    return model, progress.history


def infer_nodes(
    features: RoadFeatures,
    model: PPO,
    config: RoadPPOConfig,
    source: int,
    target: int,
    start_minute: int,
    budget_min: float,
    *,
    stochastic: bool = False,
) -> RoadEnv:
    """Follow policy edges between node positions without a path fallback.

    Args:
        features: Shared full-graph features for local observations.
        model: Loaded or newly trained SB3 PPO model.
        config: Reward rules and maximum trip edge count.
        source: Starting node row position.
        target: Distinct destination node row position.
        start_minute: Departure minute after midnight, in [0, 1440).
        budget_min: Positive finite travel-time budget in minutes.
        stochastic: Whether to sample actions instead of taking their mode.

    Returns:
        Trip containing actual traversed edges and success/failure state. A
        failed or initially all-masked trip is returned for evaluation; its
        path is never replaced with a Dijkstra route.

    Raises:
        ValueError: Endpoints, clock, budget or a chosen action are invalid.
        InterruptedError: The graph's cancellation callback requests a stop.
    """
    env = RoadEnv(features, source, target, start_minute, budget_min, config)
    while not env.done and env.observe()["mask"].any():
        features.graph.check_cancelled()
        action, _ = model.predict(features.pack(env), deterministic=not stochastic)
        env.step(int(action))
    return env


def infer(
    features: RoadFeatures,
    model: PPO,
    config: RoadPPOConfig,
    start: str,
    end: str,
    start_minute: int,
    budget_min: float,
) -> RouteSnapshot:
    """Build the shared route snapshot for a successful configured-point trip.

    Args:
        features: Graph features whose snapped point IDs identify endpoints.
        model: SB3 road policy used to choose every traversed edge.
        config: Routing settings, including the trip edge limit.
        start: Configured starting hub or POI ID.
        end: Distinct configured destination ID.
        start_minute: Departure minute after midnight, in [0, 1440).
        budget_min: Positive finite travel-time budget in minutes.

    Returns:
        Route snapshot containing actual edge geometry, road instructions,
        unique passed POIs, score, minutes, kilometres and width coverage.
        Distinct IDs sharing a snapped node produce a zero-edge snapshot.

    Raises:
        ValueError: Endpoints, departure or budget are invalid, or the policy
            fails to reach the destination within the budget/edge limit.
        InterruptedError: The graph's cancellation callback requests a stop.
    """
    graph = features.graph
    if (
        start == end
        or not np.isfinite(budget_min)
        or budget_min <= 0
        or not isinstance(start_minute, int)
        or not 0 <= start_minute < 1440
    ):
        raise ValueError("Invalid PPO endpoints, departure or time budget")
    if start not in graph.snapped or end not in graph.snapped:
        raise ValueError("PPO endpoints must be within 300 m of a road node")
    source, target = graph.snapped[start], graph.snapped[end]
    if source == target:
        return cast(
            "RouteSnapshot",
            graph.render_path([start, end], [], start_minute, budget_min * 60),
        )
    env = infer_nodes(features, model, config, source, target, start_minute, budget_min)
    if not env.success:
        raise ValueError(
            "PPO did not reach the destination within the budget/step limit"
        )
    return cast(
        "RouteSnapshot",
        graph.render_path([start, end], env.path, start_minute, budget_min * 60),
    )


def evaluate(
    features: RoadFeatures,
    model: PPO,
    config: RoadPPOConfig,
    scenarios: Iterable[RoadScenario],
) -> EvaluationReport:
    """Evaluate fixed node-position scenarios using deterministic inference.

    Args:
        features: Shared full-graph features and return-time cache.
        model: SB3 road policy to evaluate without training updates.
        config: Routing settings, including the maximum trip edge count.
        scenarios: Cases of source node, target node, departure minute and
            travel-time budget in minutes.

    Returns:
        Completion counts and one result per case. Failed trips have a null
        score; duration, distance and width exposure describe their partial
        edge path, not a completed route.

    Raises:
        ValueError: A scenario's nodes, departure minute or budget are invalid.
        InterruptedError: The graph's cancellation callback requests a stop.
    """
    results: list[ScenarioResult] = []
    for source, target, minute, budget in scenarios:
        env = infer_nodes(features, model, config, source, target, minute, budget)
        lengths, widths = (
            features.graph.lengths[env.path],
            features.graph.widths[env.path],
        )
        results.append(
            {
                "source": source,
                "target": target,
                "departure_minute": minute,
                "budget_min": budget,
                "success": env.success,
                "score": features.graph.mask_score(env.collected)
                if env.success
                else None,
                "duration_min": env.elapsed_s / 60,
                "steps": env.steps,
                "distance_km": float(lengths.sum()) / 1000,
                "narrow_m": float(lengths[np.isfinite(widths) & (widths < 5.5)].sum()),
                "unknown_width_m": float(lengths[np.isnan(widths)].sum()),
            }
        )
    return {
        "scenarios": results,
        "completed": sum(row["success"] for row in results),
        "count": len(results),
    }
