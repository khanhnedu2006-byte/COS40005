"""Train the OD-matrix PPO routing policy."""

import argparse

from ppo_routes import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matrix", default="data/final/od_matrix.csv")
    args, remaining = parser.parse_known_args()
    main(["--matrix", args.matrix, "train", *remaining])
