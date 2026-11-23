"""Train, evaluate or infer an SB3 PPO policy on the complete road graph.

Run from the repository root after editable installation. Progress goes to
stderr; stdout and --report contain the final JSON report. See
``documentation/ppo-road.md`` for the data flow and checkpoint semantics.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from busroute.routing.ppo_road import (
    EvaluationReport,
    RoadFeatures,
    RoadPPOConfig,
    checkpoint_path,
    evaluate,
    infer,
    load_policy,
    read_checkpoint,
    sample_env,
    train,
)
from busroute.routing.road_graph import RoadGraph, configured_points
from busroute.utils.timeslots import to_minutes

if TYPE_CHECKING:
    from busroute.routing.ppo_road import _RoadModelState
    from Web.types import RouteSnapshot

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Build the road PPO train/evaluate/infer command-line interface.

    Returns:
        Argument parser with data-path flags and three subcommands. Training
        overrides default to None so resume can inherit non-default settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default=str(ROOT / "data/stage1/nodes_base.parquet"))
    parser.add_argument("--edges", default=str(ROOT / "data/stage1/edges_base.parquet"))
    parser.add_argument(
        "--widths", default=str(ROOT / "data/final/duong_pho_clean.csv")
    )
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train or resume standard SB3 PPO")
    training.add_argument("--checkpoint", default="checkpoints/ppo_road.zip")
    training.add_argument(
        "--resume", help="Continue an SB3 .zip checkpoint with its saved settings"
    )
    training.add_argument(
        "--steps",
        type=int,
        default=200_000,
        help="Additional interactions; SB3 rounds up to complete rollouts",
    )
    # None distinguishes an explicit override from a default. Resume inherits
    # the saved configuration, including non-default architecture/reward flags.
    for flag, field, kind, help_text in (
        (
            "rollout-steps",
            "rollout_steps",
            int,
            "Transitions per PPO update (default 256)",
        ),
        (
            "batch-size",
            "minibatch_size",
            int,
            "Minibatch size; must divide rollout steps (default 64)",
        ),
        ("epochs", "epochs", int, "Optimizer passes per rollout (default 4)"),
        ("hidden-dim", "hidden_dim", int, "GraphSAGE/actor/critic width (default 32)"),
        ("seed", "seed", int, "Fresh-training seed (default 0)"),
        ("threads", "threads", int, "Torch CPU threads (default 4)"),
        (
            "max-episode-steps",
            "max_episode_steps",
            int,
            "Edge limit before a trip fails (default 2048)",
        ),
        (
            "curriculum-steps",
            "curriculum_steps",
            int,
            "Steps to grow trip separation (default 50000)",
        ),
        ("lr", "lr", float, "Learning rate (default 0.0003)"),
        (
            "time-penalty",
            "time_penalty",
            float,
            "Reward deducted per travel minute (default 0.01)",
        ),
        (
            "narrow-penalty",
            "narrow_penalty",
            float,
            "Extra deduction per known narrow-road minute (default 0.05)",
        ),
        (
            "arrival-bonus",
            "arrival_bonus",
            float,
            "Reward for reaching the destination (default 10)",
        ),
        (
            "failure-penalty",
            "failure_penalty",
            float,
            "Reward deducted for a failed trip (default 20)",
        ),
    ):
        training.add_argument(
            "--" + flag, dest=field, type=kind, default=None, help=help_text
        )
    training.add_argument(
        "--eval-count",
        type=int,
        default=20,
        help="Fixed held-out trips for fresh training; ignored on resume",
    )
    training.add_argument(
        "--report", help="Write training and held-out evaluation JSON"
    )
    evaluation = commands.add_parser(
        "evaluate", help="Evaluate the checkpoint's held-out trips"
    )
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--report")
    inference = commands.add_parser(
        "infer", help="Follow a trained policy between configured points"
    )
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--start", required=True)
    inference.add_argument("--end", required=True)
    inference.add_argument("--departure", default="08:00")
    inference.add_argument("--budget-min", type=float, default=240)
    return parser


def training_config(
    args: argparse.Namespace,
    saved: Mapping[str, int | float] | None = None,
) -> RoadPPOConfig:
    """Combine fresh or saved defaults with explicit training CLI overrides.

    Args:
        args: Parsed train arguments; steps is the additional interaction budget.
        saved: Optional checkpoint configuration; None uses fresh defaults.

    Returns:
        Validated RoadPPOConfig. Conflicting resume overrides are subsequently
        rejected by train, which compares every setting except total_steps.

    Raises:
        ValueError: The resulting PPO dimensions or coefficients are invalid.
    """
    values: dict[str, Any] = dict(saved) if saved else asdict(RoadPPOConfig())
    for field in values:
        value = args.steps if field == "total_steps" else getattr(args, field, None)
        if value is not None:
            values[field] = value
    config = RoadPPOConfig(**values)
    config.validate()
    return config


def train_report(args: argparse.Namespace, features: RoadFeatures) -> EvaluationReport:
    """Choose held-out cases, run learning, and evaluate the resulting model.

    Args:
        args: Parsed train arguments, including checkpoint/resume/report settings.
        features: Loaded full-graph features shared by sampling and learning.

    Returns:
        Evaluation report with a training section describing history, settings,
        graph size, backend and requested interactions. Progress is emitted to
        stderr; file/stdout output is left to main.

    Raises:
        ValueError: Evaluation count, settings, sampling or checkpoint is invalid.
        OSError: Checkpoint reading or writing fails.
        InterruptedError: The graph's cancellation callback requests a stop.
    """
    metadata = read_checkpoint(args.resume, features) if args.resume else None
    config = training_config(args, metadata["config"] if metadata else None)
    if args.eval_count < 0:
        raise ValueError("eval-count must be non-negative")
    heldout = metadata["heldout"] if metadata else []
    if metadata is None:
        rng = np.random.default_rng(config.seed + 1_000_000)
        for _ in range(args.eval_count):
            env = sample_env(features, rng, config, config.curriculum_steps, heldout)
            heldout.append(
                (env.source, env.target, env.start_minute, env.budget_s / 60)
            )
    model, history = train(
        features,
        config,
        args.checkpoint,
        resume=args.resume,
        heldout=heldout,
        callback=lambda row: print(json.dumps(row), file=sys.stderr, flush=True),
    )
    report = evaluate(features, model, config, heldout)
    report["training"] = {
        "history": history,
        "config": asdict(config),
        "checkpoint": str(checkpoint_path(args.checkpoint)),
        "nodes": len(features.graph.nodes),
        "edges": len(features.graph.edges),
        "backend": "sb3",
        "device": "cpu",
        "requested_steps": config.total_steps,
    }
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch a road PPO command and emit its final JSON result.

    Args:
        argv: Command-line tokens; None reads the process argument vector.

    Raises:
        SystemExit: Argparse handles help/usage, or a file, value or metadata
            error exits with status 2 and a readable stderr message.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    report: EvaluationReport | RouteSnapshot
    try:
        checkpoint_path(args.checkpoint)
        if args.command == "train" and args.resume:
            checkpoint_path(args.resume)
        graph = RoadGraph.from_files(
            args.nodes, args.edges, args.widths, configured_points(ROOT)
        )
        features = RoadFeatures(graph)
        if args.command == "train":
            report = train_report(args, features)
        else:
            model, config = load_policy(args.checkpoint, features)
            if args.command == "infer":
                if not re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", args.departure):
                    raise ValueError("Departure must be a valid HH:MM time")
                report = infer(
                    features,
                    model,
                    config,
                    args.start,
                    args.end,
                    to_minutes(args.departure),
                    args.budget_min,
                )
            else:
                heldout = cast("_RoadModelState", model).road_metadata["heldout"]
                if not heldout:
                    raise ValueError("Checkpoint has no held-out evaluation scenarios")
                report = evaluate(features, model, config, heldout)
        if getattr(args, "report", None):
            path = Path(args.report)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
