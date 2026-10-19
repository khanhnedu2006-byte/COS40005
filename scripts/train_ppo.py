"""Huấn luyện PPO + GNN trên dataset đồ thị đã xuất."""
import argparse

import pandas as pd

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/processed")
    args = ap.parse_args()
    nodes = pd.read_parquet(f"{args.graph}/nodes.parquet")
    edges = pd.read_parquet(f"{args.graph}/edges.parquet")
    print("Loaded", len(nodes), "nodes,", len(edges), "edges")
    # TODO: dựng TourRouteEnv, GraphEncoder, PolicyValueHead rồi gọi busroute.routing.ppo_train.train
