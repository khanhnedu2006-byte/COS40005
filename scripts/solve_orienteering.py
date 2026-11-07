"""Run ALNS, VNS, or a trained PPO policy on the project's OD matrix."""
from __future__ import annotations

import argparse
import json

from busroute.routing.metaheuristics import ALNS, VNS, Algorithm
from busroute.routing.orienteering import ODMatrix
from busroute.utils.timeslots import to_minutes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="data/final/od_matrix.csv")
    parser.add_argument("--algorithm", choices=("alns", "vns", "ppo"), default="alns")
    parser.add_argument("--checkpoint", help="Trained policy checkpoint (required for PPO)")
    parser.add_argument("--start", required=True, help="Start point ID, e.g. H01")
    parser.add_argument("--end", required=True, help="Destination point ID, e.g. H02")
    parser.add_argument("--departure", default="08:00", help="Local departure time HH:MM")
    parser.add_argument("--budget-min", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.algorithm == "ppo" and not args.checkpoint:
        parser.error("--checkpoint is required for PPO")
    try:
        parts = args.departure.split(":")
        if len(parts) != 2 or not all(part.isdigit() and len(part) == 2 for part in parts):
            raise ValueError("Departure must be HH:MM")
        if int(parts[0]) >= 24 or int(parts[1]) >= 60:
            raise ValueError("Departure must be a valid time in HH:MM")
        matrix = ODMatrix.from_csv(args.matrix)
        if args.algorithm == "ppo":
            # Keep PPO's ML dependencies optional for the metaheuristic solvers.
            from busroute.routing.ppo_od import Features, PPOSearch, load_coordinates, load_policy

            algorithm: Algorithm = PPOSearch(
                load_policy(args.checkpoint, args.matrix),
                Features(matrix, load_coordinates(args.matrix)),
            )
        elif args.algorithm == "alns":
            algorithm = ALNS(iterations=args.iterations, seed=args.seed)
        else:
            algorithm = VNS(iterations=args.iterations, seed=args.seed)
        result = algorithm(matrix, args.start, args.end, to_minutes(args.departure),
                           args.budget_min * 60)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
