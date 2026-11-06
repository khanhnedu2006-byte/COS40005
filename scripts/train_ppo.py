"""Huấn luyện PPO + GNN trên dataset đồ thị đã xuất."""

import argparse
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/processed")
    args = ap.parse_args()
    graph = Path(args.graph)
    missing = [name for name in ("nodes.parquet", "edges.parquet") if not (graph / name).is_file()]
    if missing:
        ap.error(
            f"Missing graph files in {graph}: {', '.join(missing)}. "
            f"Generate them with: python scripts/build_graph.py --out {graph}. "
            "For the implemented OD-matrix PPO trainer, use scripts/ppo_routes.py train."
        )
    nodes = pd.read_parquet(graph / "nodes.parquet")
    edges = pd.read_parquet(graph / "edges.parquet")
    print("Loaded", len(nodes), "nodes,", len(edges), "edges")
    print("Graph-based PPO training is not implemented; use scripts/ppo_routes.py train.")
    # TODO: dựng TourRouteEnv, GraphEncoder, PolicyValueHead rồi gọi busroute.routing.ppo_train.train
