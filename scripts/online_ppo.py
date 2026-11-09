"""Learn a PPO policy on simulated attempts at one requested route."""

import argparse
import json

from busroute.routing.orienteering import ODMatrix
from busroute.routing.ppo_od import Features, OnlinePPOSearch, PPOConfig, load_coordinates
from busroute.utils.timeslots import to_minutes
from ppo_progress import PPOProgress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="data/final/od_matrix.csv")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--departure", default="08:00")
    parser.add_argument("--budget-min", type=float, required=True)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    parts = args.departure.split(":")
    if (len(parts) != 2 or any(len(p) != 2 or not p.isdigit() for p in parts)
            or int(parts[0]) >= 24 or int(parts[1]) >= 60):
        parser.error("departure must be a valid HH:MM time")
    try:
        matrix = ODMatrix.from_csv(args.matrix)
        features = Features(matrix, load_coordinates(args.matrix))
        config = PPOConfig(total_steps=args.steps, rollout_steps=args.rollout_steps,
                           minibatch_size=args.minibatch_size, epochs=args.epochs,
                           hidden_dim=args.hidden_dim, seed=args.seed)
        solver = OnlinePPOSearch(features, config, callback=PPOProgress())
        route = solver(matrix, args.start, args.end, to_minutes(args.departure), args.budget_min * 60)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(route.to_dict(), indent=2))


if __name__ == "__main__":
    main()
