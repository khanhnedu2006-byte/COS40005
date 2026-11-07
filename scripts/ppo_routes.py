"""Train, infer, and evaluate PPO routes over an OD matrix."""

from __future__ import annotations

import argparse
import json

from busroute.routing.orienteering import ODMatrix
from busroute.routing.od_evaluation import evaluate, greedy_route, shortest_route
from busroute.routing.metaheuristics import ALNS, VNS
from busroute.routing.ppo_od import (
    Features,
    PPOConfig,
    PPOSearch,
    load_coordinates,
    load_policy,
    scenarios,
    train,
)
from busroute.utils.timeslots import to_minutes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="data/final/od_matrix.csv")
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument(
        "--matrix",
        default=argparse.SUPPRESS,
        help="OD matrix CSV (default: data/final/od_matrix.csv)",
    )
    training.add_argument(
        "--checkpoint",
        default="checkpoints/ppo.zip",
        help="Output checkpoint path (default: checkpoints/ppo.zip)",
    )
    training.add_argument("--steps", type=int, default=200000)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--lr", type=float, default=0.0003)
    training.add_argument("--gamma", type=float, default=0.99)
    training.add_argument("--clip-eps", type=float, default=0.2)
    training.add_argument("--hidden-dim", type=int, default=64)
    training.add_argument("--rollout-steps", type=int, default=256)
    training.add_argument("--epochs", type=int, default=4)
    training.add_argument("--minibatch-size", type=int, default=32)
    training.add_argument("--eval-cases", type=int, default=20)
    inference = sub.add_parser("infer")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--start", required=True)
    inference.add_argument("--end", required=True)
    inference.add_argument("--departure", default="08:00")
    inference.add_argument("--budget-min", type=float, required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--iterations", type=int, default=100)
    evaluation.add_argument(
        "--output", help="Write the complete JSON report to this path"
    )
    args = parser.parse_args(argv)
    matrix = ODMatrix.from_csv(args.matrix)
    if args.command == "train":
        cfg = PPOConfig(
            total_steps=args.steps,
            seed=args.seed,
            lr=args.lr,
            gamma=args.gamma,
            clip_eps=args.clip_eps,
            hidden_dim=args.hidden_dim,
            rollout_steps=args.rollout_steps,
            epochs=args.epochs,
            minibatch_size=args.minibatch_size,
        )
        result = train(matrix, args.matrix, cfg, args.checkpoint, args.eval_cases)
    else:
        model = load_policy(args.checkpoint, args.matrix)
        features = Features(matrix, load_coordinates(args.matrix))
        solver = PPOSearch(model, features)
        if args.command == "infer":
            parts = args.departure.split(":")
            if (
                len(parts) != 2
                or any(not p.isdigit() or len(p) != 2 for p in parts)
                or int(parts[0]) >= 24
                or int(parts[1]) >= 60
            ):
                parser.error("departure must be a valid HH:MM time")
            try:
                route = solver(
                    matrix,
                    args.start,
                    args.end,
                    to_minutes(args.departure),
                    args.budget_min * 60,
                )   
            except ValueError as exc:
                parser.exit(2, f"error: {exc}\n")
            result = route.to_dict()
        else:
            cases = scenarios(
                matrix,
                model.routing_eval_count,
                model.routing_config["seed"] + 1_000_000,
            )
            solvers = {
                "ppo": solver,
                "shortest": shortest_route,
                "greedy": greedy_route,
                "alns": ALNS(
                    iterations=args.iterations, seed=model.routing_config["seed"]
                ),
                "vns": VNS(
                    iterations=args.iterations, seed=model.routing_config["seed"]
                ),
            }
            result = evaluate(matrix, cases, solvers)
    output = json.dumps(result, indent=2)
    if args.command == "evaluate" and args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
