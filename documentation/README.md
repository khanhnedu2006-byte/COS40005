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
time and travel budget, then run the search. ALNS, VNS and Ant Colony also accept
iteration and random-seed inputs. Dijkstra uses neither of those settings.

Use `--no-browser`, `--port 8001`, or
`--matrix data/final/od_matrix_mock.csv` after `-m Web` when needed. The checkout's
configuration, data files and optional `web` dependencies must be available.
Tiles and labels require internet access; the solver uses the local OD matrix.

## Reading guide

- [Algorithms](algorithms.md): common evaluation rules; ALNS, VNS, Ant Colony,
  Dijkstra, the greedy baseline and the existing PPO approach.
- [Visualization approaches](visualization.md): static Folium, native desktop
  plotting, and the current interactive browser map, including their tradeoffs.
- [Web application](web-application.md): folder structure, typed contracts,
  event protocol, process lifecycle, geometry caching and HTTP endpoints.
- [Development and verification](development.md): type checks, Google-style
  docstrings, regression tests, browser checks and known limitations.

The web application supports four solvers: `alns`, `vns`, `aco`, and `dijkstra`.
PPO remains available through the project's existing command-line tools.
The original solver CLI can still be used, for example:

```bash
python scripts/solve_orienteering.py --algorithm aco --start H01 --end H02 --departure 08:00 --budget-min 240 --iterations 100 --seed 42
```

Road lines are reconstructed on OSM. Their shape is illustrative: the OD matrix,
not the displayed OSM path, supplies each result's score and travel time.
