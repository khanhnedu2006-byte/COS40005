# COS40005 – Vehicle-Aware Route Planning & Driver Facial Analysis

Routing for **45-seat tour buses** across Da Nang city (including former Quảng Nam / Hội An), followed by driver facial analysis from cabin video.

- Project context for teammates / AI assistants: [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md)
- Architecture & source layout: [`documentation/ARCHITECTURE.md`](documentation/ARCHITECTURE.md)

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .
```

## Run the pipeline
```bash
python scripts/normalize_dataset.py --src <raw exports> --out dataset     # Step 0: unify raw exports
python scripts/run_clean_gps.py --src dataset --out data/interim/gps   # Step 1
python scripts/build_graph.py   --out data/processed                       # Steps 2–6
pytest                                                                     # Tests
```

`build_graph.py` generates road graph files in its output directory.

## Conventions
- Variable / function / file names in **English**.
- All bus restrictions are **hard-coded** in `src/busroute/config/` – OSM is used only as the base map.
- Never commit raw GPS logs or any driver video (see `.gitignore`).

## Orienteering and shortest-route baseline

The OD-matrix solvers maximise the scores of unique POIs between a fixed start
and end, subject to a travel-time budget. Run from the repository root after
installing the package:

```bash
python scripts/solve_orienteering.py --algorithm alns --start H01 --end H02 --departure 08:00 --budget-min 240 --iterations 100 --seed 42
python scripts/solve_orienteering.py --algorithm vns --start H01 --end H02 --departure 08:00 --budget-min 240 --iterations 100 --seed 42
python scripts/solve_orienteering.py --algorithm aco --start H01 --end H02 --departure 08:00 --budget-min 240 --iterations 100 --seed 42
python scripts/solve_orienteering.py --algorithm dijkstra --start H01 --end H02 --departure 08:00 --budget-min 240
```

Use `--matrix data/final/od_matrix_mock.csv` for a mock-data smoke test. Output is
JSON with `route`, `score`, `duration_min` and `collected` for all solvers.
Add `--verbose` to write JSON-line progress to stderr while keeping the route
JSON on stdout. ALNS, VNS and ACO report the initial solution and best score
and duration after each iteration. Dijkstra reports search start, every 100
expanded labels, and completion (including unsuccessful searches). Without
`--verbose`, these progress records are suppressed. For example:

```bash
python scripts/solve_orienteering.py --algorithm alns --start H01 --end H02 --budget-min 240 --iterations 100 --verbose 2> search.jsonl
```

To open a native OSMnx map window, use the same arguments with
`scripts/visualize_orienteering.py`:

```bash
python -m pip install -e ".[visualization]"
python scripts/visualize_orienteering.py --algorithm alns --start H01 --end H02 --departure 08:00 --budget-min 240 --iterations 100 --seed 42
```

Tkinter and a graphical desktop are required (`python3-tk` on Ubuntu/Debian).
The window starts the solver automatically; it has pan/zoom, Pause/Resume and
animation speed controls. Close it to stop. The completed route is printed as
JSON and the window stays open for inspection. ALNS and VNS animate selected
removal, insertion and local-search actions, plus acceptance decisions. ACO and
Dijkstra show their available progress and the final route.

Roads use the existing cache in `configs/settings.yaml`; a missing graph is
downloaded through the existing OSM pipeline. Blue shows the best route, green
the current route and dashed orange the candidate. Paths are reconstructed on
the directed OSM graph by road length; reported scores and times still come
from the OD matrix. These paths can differ from the original provider routes
and do not certify bus restrictions. Missing connections or stops more than
300 metres from the graph are reported rather than drawn as road paths.

### Interactive browser map

The application now lives in the root-level [`Web/`](Web/) package. See
[`documentation/`](documentation/README.md) for algorithms, visualization
approaches, API contracts, type annotations and verification instructions.

For a street map with location inputs and a final-route guide, use the steps
below. Only Python and pip are needed for installation; `uv` is not required.
Use Python 3.10 or newer and open a terminal in the repository root (the folder
containing `pyproject.toml`). Keep the checkout's `configs/` and `data/` folders
available, including the base node/edge Parquet files and road-width CSV.

```bash
python -m venv .venv
```

Activate the virtual environment using the command for your terminal:

- Windows Command Prompt: `.venv\Scripts\activate.bat`
- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- Linux/macOS: `source .venv/bin/activate`

If `python` is unavailable on Linux/macOS, use `python3 -m venv .venv` to create
the environment. If you already created an environment during Setup, activate
it and skip creating another one.

Install the project with its web dependencies, then launch it:

```bash
python -m pip install -e ".[web]"
python -m Web
```

This starts a local server at `http://127.0.0.1:8000` and opens your browser.
If the browser does not open, visit that address manually. Keep the terminal
running while using the interface. On subsequent launches, activate the same
environment and run `python -m Web` from the repository root; installation only
needs repeating when dependencies change.

No Node.js, frontend build, map account or API key is required.
MapLibre GL JS 5.24.0 is bundled locally; the OpenFreeMap
basemap, labels and glyphs require internet access and a WebGL-capable browser.

Choose endpoints from the Start and Destination dropdowns, or click a map marker.
Select **Dijkstra**, **ALNS**, **VNS**, **ACO**, or **PPO**, set a **time budget in minutes**,
and choose Find route. ALNS, VNS and ACO also accept iterations and a random seed; PPO accepts a checkpoint path (defaulting to `checkpoints/ppo_reduced.zip`).
Defaults are 240 minutes, departure 08:00, 100 iterations and seed 0. Cancel stops the isolated solver
process. Only the final result is displayed.

The web solvers use `data/stage1_reduced/` when available (or `data/stage1/*_base.parquet`),
`data/final/duong_pho_clean.csv`, and configured POIs/hubs. Dijkstra, ALNS, VNS and ACO build directed
POI connections on the road graph (`via_nodes` restored when using the reduced graph). ALNS and VNS
insert POI detours such as A → P → B, maximizing unique POI score with ties broken by 2× narrow-road-weighted distance, estimated time, and physical distance, while ACO optimizes PPO's reward/penalty objective. Scores credit each POI once when a route visits the graph node it snaps to (matching PPO).
Dijkstra supplies the fastest-estimated-time baseline.

To benchmark all 5 algorithms across the 3 execution modes (`poi`, `street-reduced`, `street-full`) from the CLI:

```bash
python scripts/evaluate_algorithms.py --mode all --start H01 --end H04 --budget-min 90 --departure 08:00 --iterations 24
```

The result includes numbered waypoints, unique POI score, estimated travel time and physical distance,
highlighted passed POIs, width coverage and estimated road directions. Clicking
a direction focuses the map at that road change. Download JSON saves the same
route geometry and metrics. With the reduced graph, `via_nodes` expands intermediate road curves back to base-node coordinates. POIs earn credit only when the route visits their snapped road node (within 300 m). Times are estimates using the historical TomTom-calibrated road-class coefficients in `configs/estimator_params.json`. Each edge time is length × seconds/metre at that edge's departure slot; the clock advances after every edge for all road solvers. The road graph retains road classes without a calibrated group using the local-road coefficients. No per-connection fixed overhead is added, matching the earlier road-edge dataset builder. Live traffic and legal bus restrictions are not modeled.

Points more than 300 m from a road node stay visible but cannot be selected as
endpoints or inserted waypoints. Their map popups explain the snapping limit.
The previous OD-matrix CLI solvers and desktop viewer remain available with
their existing minute-based inputs.

To use another port without opening a browser:

```bash
python -m Web --port 8001 --no-browser
```

The server retains the current run until replaced or restarted. Only one run
can be active at a time. Ctrl+C stops the server and worker. Basemap tiles need
internet access; use Retry map if tiles fail. API tests require the test extra:
`python -m pip install -e ".[web,test]"`.

`--iterations` and `--seed` configure ALNS, VNS and ant colony optimisation
(ACO). The Dijkstra-style baseline seeks a short feasible route,
without trying to maximise POI score; it uses time-dependent arrival-minute
labels and does not model waiting, so it is not guaranteed to find the exact
shortest route when availability or travel times change between slots.

### Legacy OD-matrix solver notes

The `duration_min` field reports travel time in minutes, matching `--budget-min`.
The mock matrix is not suitable for reporting real results.

In Python, construct `ALNS`, `VNS`, or `AntColony` and call the instance. Each implements the
`Algorithm` protocol and return a typed `RouteResult`:

```python
from busroute.routing.metaheuristics import ALNS, Algorithm
from busroute.routing.orienteering import ODMatrix

matrix = ODMatrix.from_csv("data/final/od_matrix.csv")
solver: Algorithm = ALNS(iterations=100, seed=42)  # Or VNS(...)
result = solver(matrix, "H01", "H02", start_minute=480, budget_s=14400)
```

`start_minute` is minutes after midnight and `budget_s` is in seconds.
`iterations` controls search effort; `seed` makes runs reproducible. Score is
maximised first, with shorter travel time breaking ties. POIs passed on a leg
and POIs selected as stops receive the same score, at most once; a POI start is
also counted. The model does not add stopping or waiting time.

Each leg uses the matrix row for its departure slot, including its feasibility
flag. The matrix cannot establish whether a bus enters a restricted area after
a slot boundary during a leg; results are approximate and do not guarantee
real-world compliance. The matrix's representative travel times are estimates,
not live traffic. Search starts with a direct route or a feasible route through
one POI. If neither exists, it reports failure even if a longer route could
exist. Both methods are heuristics and do not guarantee an optimal result.
