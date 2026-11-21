# Route planning and visualization approaches

This folder explains the route-search implementations and the visualization
approaches developed around them. It supplements the existing project-wide
[architecture](../docs/ARCHITECTURE.md) and [project context](../PROJECT_CONTEXT.md).

## Start the browser application

Run from the repository root:

```bash
uv run --extra web python -m Web
```

The server opens `http://127.0.0.1:8000`. Select endpoints, an algorithm, departure
time and travel budget, then run the search. ALNS, VNS and ACO also accept
iteration and random-seed inputs. PPO uses a saved road SB3 checkpoint.
Only the final path is visualized, with an ordered, map-linked route guide.

Use `--no-browser` or `--port 8001` after `-m Web` when needed. The checkout's
configuration, data files and optional `web` dependencies must be available.
Tiles and labels require internet access; the solvers use the local road graph.

## Reading guide

- [Algorithms](algorithms.md): common evaluation rules; ALNS, VNS, Ant Colony,
  Dijkstra, Waypoint PPO and Edge PPO.
- [Road PPO training](ppo-road.md): SB3 ownership, commented module walkthrough,
  graph observations, masking, rewards, checkpoints and performance limits.
- [Visualization approaches](visualization.md): static Folium, native desktop
  plotting, and the current interactive browser map, including their tradeoffs.
- [Web application](web-application.md): folder structure, typed contracts,
  final-result polling, process lifecycle, geometry caching and HTTP endpoints.
- [Development and verification](development.md): type checks, Google-style
  docstrings, regression tests, browser checks and known limitations.

The web application supports `alns`, `vns`, `aco`, `ppo`, `ppo_waypoint` and `dijkstra`.
The OD-matrix algorithms remain available through the separate command-line tools.
The original solver CLI can still be used, for example:

```bash
python scripts/solve_orienteering.py --algorithm aco --start H01 --end H02 --departure 08:00 --budget-min 240 --iterations 100 --seed 42
```

Web geometry, POI score and travel time derive from the actual selected road
edges. The separate OD-matrix CLI uses matrix scores/times and illustrative OSM
reconstruction, as described in the visualization notes.
