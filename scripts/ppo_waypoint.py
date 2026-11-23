"""Train, run or benchmark Waypoint PPO on the full road graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import cast

from busroute.routing import ppo_road, ppo_waypoint
from busroute.routing.road_graph import (
    NoRoadRouteError,
    RoadGraph,
    RoadProblem,
    configured_points,
)
from busroute.routing.road_metaheuristics import search
from busroute.routing.waypoint_env import BENCHMARK_CASES, WaypointFeatures
from busroute.utils.timeslots import to_minutes

ROOT = Path(__file__).resolve().parents[1]
WAYPOINT_CHECKPOINT = ROOT / "checkpoints/ppo_waypoint.zip"
EDGE_CHECKPOINT = ROOT / "checkpoints/ppo_road.zip"
ALGORITHMS = ("ppo_waypoint", "ppo", "dijkstra", "alns", "vns", "aco")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default=ROOT / "data/stage1/nodes_base.parquet")
    parser.add_argument("--edges", default=ROOT / "data/stage1/edges_base.parquet")
    parser.add_argument("--widths", default=ROOT / "data/final/duong_pho_clean.csv")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train a fresh waypoint model")
    training.add_argument("--steps", type=int, default=100_000)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--threads", type=int, default=4)
    training.add_argument("--rollout-steps", type=int, default=256)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--epochs", type=int, default=4)
    inference = commands.add_parser(
        "infer", help="Generate a trip with the saved policy"
    )
    inference.add_argument("--start", required=True)
    inference.add_argument("--end", required=True)
    inference.add_argument("--departure", default="08:00")
    inference.add_argument("--budget-min", type=float, default=240)
    comparison = commands.add_parser(
        "benchmark", help="Compare solvers on the six held-out trips"
    )
    comparison.add_argument("--edge-checkpoint", default=EDGE_CHECKPOINT)
    comparison.add_argument(
        "--report", type=Path, default=ROOT / "results/ppo_waypoint_benchmark.json"
    )
    comparison.add_argument("--iterations", type=int, default=100)
    comparison.add_argument("--seed", type=int, default=0)
    for command in (training, inference, comparison):
        command.add_argument("--checkpoint", default=WAYPOINT_CHECKPOINT)
    return parser


def progress(record):
    print(json.dumps(record), file=sys.stderr, flush=True)


def benchmark(graph, features, args):
    """Evaluate the same graph, objective and trip inputs for every solver."""
    if args.iterations < 0:
        raise ValueError("Iterations must be non-negative")
    waypoint_model = ppo_waypoint.load_policy(args.checkpoint, features)
    edge_features = ppo_road.RoadFeatures(graph)
    edge_model, edge_config = ppo_road.load_policy(args.edge_checkpoint, edge_features)
    cases = []
    totals = dict.fromkeys(ALGORITHMS, 0)
    for start, end, minute, budget in BENCHMARK_CASES:
        scenario = (start, end, minute, budget)
        results = []
        heuristic_scores: list[float] = []
        waypoint_score = None
        for name in ALGORITHMS:
            began = perf_counter()
            snapshot, error = None, None
            try:
                if name == "ppo_waypoint":
                    snapshot = ppo_waypoint.infer(
                        WaypointFeatures(graph), waypoint_model, scenario
                    )
                elif name == "ppo":
                    env = ppo_road.infer_nodes(
                        edge_features,
                        edge_model,
                        edge_config,
                        graph.snapped[start],
                        graph.snapped[end],
                        minute,
                        budget,
                    )
                    if env.success:
                        snapshot = graph.render_path(
                            [start, end], env.path, minute, budget * 60
                        )
                    else:
                        error = "Edge PPO did not reach the destination"
                else:
                    problem = RoadProblem(
                        graph,
                        start,
                        end,
                        budget,
                        start_minute=minute,
                        all_pois=name != "dijkstra",
                    )
                    route = (
                        problem.direct
                        if name == "dijkstra"
                        else search(
                            problem,
                            name,
                            iterations=args.iterations,
                            seed=args.seed,
                        )
                    )
                    snapshot = graph.snapshot(problem, route)
            except NoRoadRouteError as exc:
                error = str(exc)
            completed = snapshot is not None
            score = snapshot["score"] if completed else None
            totals[name] += completed
            if name == "ppo_waypoint":
                waypoint_score = score
            if name in ("alns", "vns", "aco") and score is not None:
                heuristic_scores.append(score)
            row = {
                "algorithm": name,
                "completed": completed,
                "error": error,
                "score": score,
                "duration_min": snapshot["duration_min"] if completed else None,
                "solver_seconds": perf_counter() - began,
                "waypoints": snapshot["route"] if completed else None,
                "collected": snapshot["collected"] if completed else None,
            }
            results.append(row)
            progress(
                {
                    "start": start,
                    "end": end,
                    "minute": minute,
                    "budget_min": budget,
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in {"waypoints", "collected"}
                    },
                }
            )
        best_heuristic = max(heuristic_scores, default=None)
        cases.append(
            {
                "start": start,
                "end": end,
                "departure": f"{minute // 60:02d}:{minute % 60:02d}",
                "budget_min": budget,
                "results": results,
                "best_heuristic_score": best_heuristic,
                "waypoint_score_gap": best_heuristic - waypoint_score
                if best_heuristic is not None and waypoint_score is not None
                else None,
            }
        )
    checkpoint = Path(args.checkpoint).resolve()
    return {
        "checkpoint": str(
            checkpoint.relative_to(ROOT)
            if checkpoint.is_relative_to(ROOT)
            else checkpoint
        ),
        "training": cast(
            ppo_waypoint.WaypointModelState, waypoint_model
        ).waypoint_metadata,
        "edge_checkpoint_steps": edge_model.num_timesteps,
        "heuristic_iterations": args.iterations,
        "heuristic_seed": args.seed,
        "runtime_scope": "Solver and per-case setup excluding shared graph and checkpoint loading",
        "completion_counts": totals,
        "cases": cases,
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        ppo_road.checkpoint_path(args.checkpoint)
        if args.command == "benchmark":
            ppo_road.checkpoint_path(args.edge_checkpoint)
        graph = RoadGraph.from_files(
            args.nodes, args.edges, args.widths, configured_points(ROOT)
        )
        features = WaypointFeatures(graph)
        if args.command == "train":
            model = ppo_waypoint.train(
                features,
                args.checkpoint,
                steps=args.steps,
                seed=args.seed,
                threads=args.threads,
                rollout_steps=args.rollout_steps,
                batch_size=args.batch_size,
                epochs=args.epochs,
                progress=progress,
            )
            output = cast(ppo_waypoint.WaypointModelState, model).waypoint_metadata
        elif args.command == "infer":
            parts = args.departure.split(":")
            if (
                len(parts) != 2
                or not all(len(p) == 2 and p.isascii() and p.isdigit() for p in parts)
                or int(parts[0]) >= 24
                or int(parts[1]) >= 60
            ):
                raise ValueError("Departure must be a valid HH:MM time")
            model = ppo_waypoint.load_policy(args.checkpoint, features)
            output = ppo_waypoint.infer(
                features,
                model,
                (args.start, args.end, to_minutes(args.departure), args.budget_min),
            )
        else:
            output = benchmark(graph, features, args)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(output, indent=2) + "\n", encoding="utf-8"
            )
        print(json.dumps(output, indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
