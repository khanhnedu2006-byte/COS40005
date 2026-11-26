# Documentation index

This folder consolidates the project's architecture, data pipeline, routing
algorithms, PPO reinforcement learning, visualization, and web-application
documentation. It supplements [`PROJECT_CONTEXT.md`](../PROJECT_CONTEXT.md)
and the root [`README.md`](../README.md).

## Start the browser application

Run from the repository root:

```bash
uv run --extra web python -m Web
```

The server opens `http://127.0.0.1:8000`. Select endpoints, an algorithm,
departure time, and travel budget, then run the search. ALNS, VNS, and ACO also
accept iteration and random-seed inputs; PPO accepts a trained `.zip` checkpoint
path (defaulting to `checkpoints/ppo_reduced.zip`). Only the final path is
visualized, with an ordered, map-linked route guide.

Use `--no-browser` or `--port 8001` after `-m Web` when needed. The checkout's
configuration, data files, and optional `web` dependencies must be available.
Tiles and labels require internet access; the solvers use the local road graph.

## Reading guide

- [Architecture](ARCHITECTURE.md): system layers, data pipeline, module layout,
  and dependency rules (plus Mermaid diagram [`architecture.mmd`](architecture.mmd)).
- [Data tasks](DATA_TASKS.md): v2 data pipeline status, TomTom OD matrix,
  central-ring restrictions, and GPS validation.
- [Arbitrary A/B routing specification](AB_ROUTING.md): `ab_estimator.py` spec,
  arbitrary A/B train/eval/demo protocol, and time-dependent features.
- [Algorithms](algorithms.md): common evaluation rules, road-graph solvers
  (Dijkstra, ALNS, VNS, ACO), and legacy OD-matrix baselines.
- [PPO routing](ppo.md): lossless graph reduction, `BusRouteEnv` MDP,
  `MaskablePPO` network architecture, episode-batch training loop, performance
  profile, benchmarks, and visualization/web inference.
- [Visualization approaches](visualization.md): static Folium, native desktop
  plotting, and the current interactive browser map, including their tradeoffs.
- [Web application and development guide](web-application.md): setup, HTTP
  contracts, typed boundaries, static checks, regression tests, browser
  verification, and known limits.

The web application supports `alns`, `vns`, `aco`, `dijkstra`, and `ppo`.
The legacy OD-matrix algorithms remain available through the separate
command-line tools, for example:

```bash
python scripts/solve_orienteering.py --algorithm aco --start H01 --end H02 --departure 08:00 --budget-min 240 --iterations 100 --seed 42
```

Web geometry, POI score, and travel time derive from the selected road edges
(with `via_nodes` expansion for PPO on the reduced graph). The separate
OD-matrix CLI uses matrix scores/times and illustrative OSM reconstruction, as
described in the visualization notes.
