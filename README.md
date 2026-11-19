# COS40005 – Vehicle-Aware Route Planning & Driver Facial Analysis

Routing for **45-seat tour buses** across Da Nang city (including former Quảng Nam / Hội An), followed by driver facial analysis from cabin video.

- Project context for teammates / AI assistants: [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md)
- Architecture & source layout: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

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
python scripts/train_ppo.py                                               # OD-matrix PPO
python scripts/ppo_road.py train                                          # Full-road-graph PPO
pytest                                                                     # Tests
```

`build_graph.py` generates road graph files in its output directory. The legacy
`train_ppo.py` uses `data/final/od_matrix.csv`. The new `ppo_road.py` trains on
the complete base graph in `data/stage1/`; see its commands below.

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
python scripts/solve_orienteering.py --algorithm ppo --checkpoint checkpoints/ppo.zip --start H01 --end H02 --departure 08:00 --budget-min 240
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
removal, insertion and local-search actions, plus acceptance decisions. ACO,
Dijkstra and PPO show their available progress and the final route.

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
Select **Dijkstra**, **ALNS**, **VNS**, **ACO**, **PPO — road edges** or **Waypoint PPO**, set a **time budget in minutes**,
and choose Find route. ALNS, VNS and ACO also accept iterations and a random seed.
Defaults are 240 minutes, departure 08:00, 100 iterations and seed 0. Cancel stops the isolated solver
process. Only the final result is displayed.
The PPO solvers each require a trained ZIP checkpoint. Waypoint PPO chooses
POIs, and the shared Dijkstra planner drives each leg while reserving enough
time to reach the destination. See the [training and benchmark commands](documentation/algorithms.md#waypoint-ppo).

The web solvers use `data/stage1/nodes_base.parquet`, `edges_base.parquet`,
`data/final/duong_pho_clean.csv`, and configured POIs/hubs. Dijkstra, ALNS, VNS and ACO build directed
POI connections on the full road graph. ALNS, VNS and ACO
insert POI detours such as A → P → B, retaining them when they improve the score
within the time budget. Scores count POIs passed along connecting roads once.
Score ties prefer lower narrow-road-weighted distance, then estimated travel time and physical distance.
Known widths below 5.5 m have a 2× cost for this tie-break; unknown widths are
unpenalized. Dijkstra supplies the fastest-estimated-time baseline.

The result includes numbered waypoints, unique POI score, estimated travel time and physical distance,
highlighted passed POIs, width coverage and estimated road directions. Clicking
a direction focuses the map at that road change. Download JSON saves the same
route geometry and metrics. The base tables do not retain road curves, so lines
between graph nodes and turn directions are approximate. POIs must be within
100 m of that approximate path to earn score; selecting a waypoint alone does
not award it. Times are estimates using the historical TomTom-calibrated road-class coefficients in `configs/estimator_params.json`. Each edge time is length × seconds/metre at that edge's departure slot; the clock advances after every edge for all four road solvers. The full graph retains road classes without a calibrated group using the local-road coefficients. No per-connection fixed overhead is added, matching the earlier road-edge dataset builder. Live traffic and legal bus restrictions are not modeled.

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

The legacy OD-matrix PPO learns by simulation for the requested route when no checkpoint is supplied;
`--ppo-steps` controls its learning budget. `--iterations` and `--seed`
configure ALNS, VNS and ant colony optimisation (ACO), while checkpoint-based PPO uses its trained
policy for inference. The Dijkstra-style baseline seeks a short feasible route,
without trying to maximise POI score; it uses time-dependent arrival-minute
labels and does not model waiting, so it is not guaranteed to find the exact
shortest route when availability or travel times change between slots.

## PPO training on the full road graph

The road pipeline now uses **Stable-Baselines3 PPO (Torch, CPU)** on all 93,751 base nodes and
230,749 directed edges. SB3 manages rollouts, GAE and optimizer updates; our
Gymnasium environment and masked Torch GraphSAGE policy retain edge-level
routing, time budgets, unique POI rewards and penalties for known widths below
5.5 m. No dense node-by-node adjacency is stored in observations.

```bash
python scripts/ppo_road.py train --steps 200000 --threads 4 --checkpoint checkpoints/ppo_road.zip --report results/ppo_road.json
python scripts/ppo_road.py evaluate --checkpoint checkpoints/ppo_road.zip --report results/ppo_road_eval.json
python scripts/ppo_road.py infer --checkpoint checkpoints/ppo_road.zip --start H01 --end H02 --departure 08:00 --budget-min 240
python scripts/ppo_road.py train --resume checkpoints/ppo_road.zip --steps 200000 --checkpoint checkpoints/ppo_road.zip
```

See the [commented pipeline walkthrough](documentation/ppo-road.md) for the
module layout, observations, rewards, curriculum, masking, checkpoint lifecycle
and performance limits. The CLI inherits saved settings on resume. SB3 rounds
the additional step budget up to whole rollouts. Twenty fixed evaluation trips
are saved by default and their ordered endpoint pairs excluded from training.
Failures have no completed-route score. Progress goes to stderr; final JSON
and optional report files contain the results.

Select **PPO** in the browser and enter the road ZIP path. Successful policy
edges produce road instructions and highlighted passed POIs. Failed inference
is reported without a shortest-path fallback. Previous SBX road ZIPs and old Torch `.pt` checkpoints need retraining;
OD-matrix checkpoints are also incompatible with road checkpoints.

A small correctness run uses `--steps 8 --rollout-steps 8 --batch-size 4
--epochs 1 --hidden-dim 8 --max-episode-steps 8 --eval-count 0`. CPU execution is
explicit; `--threads` defaults to four. Embeddings are cached during rollouts
and inference, then recomputed with gradients for each training minibatch.
The [guide's CPU smoke comparison](documentation/ppo-road.md#cpu-performance-and-thread-controls)
measured about 6 seconds for this implementation versus 16 seconds for the
previous SBX implementation on a 32-interaction run. This is an implementation
smoke measurement, not a convergence or long-run speed guarantee.

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

## PPO routing on the OD matrix

Install the optional legacy dependencies with `pip install -e ".[legacy-od-ppo]"`.
This OD-only PPO uses SBX's JAX
implementation, Gymnasium, and a custom masked GraphSAGE actor. The OD
pipeline reads `data/final/od_matrix.csv` by default. Pass `--matrix PATH` to
`train_ppo.py`, or before the subcommand in `ppo_routes.py`, to use another matrix.
To learn while solving a specific request, without a checkpoint:

```bash
python scripts/online_ppo.py --start H01 --end H02 --departure 08:00 --budget-min 240 --steps 2048
```

PPO repeatedly tries the requested scenario in the OD-matrix simulator, updates
its policy after each rollout of 256 steps, and then returns the deterministic
policy's route. The step budget is rounded up to complete rollouts; more steps
take longer and do not guarantee a better route. This is simulated learning,
not live traffic feedback. The CLI starts a fresh policy each time; in Python,
reuse `OnlinePPOSearch(features, PPOConfig(...))` across calls to retain and
continue updating the same in-memory policy. It is lost when the process exits.
The dedicated script prints the final route as JSON on stdout and emits JSON-line
progress on stderr at the end of each rollout: steps, completed policy updates,
number of completed episodes, mean episode reward, and success rate. Rollout
events occur before the next policy update; the final event includes that update.
When no episode finishes in a rollout, reward and success rate are null. Capture
the log with `2> ppo-training.jsonl` or view and save it with
`2> >(tee ppo-training.jsonl >&2)` in Bash. Checkpoint training with
`train_ppo.py` also emits progress on stderr.
For a separately trained reusable checkpoint, for example:

```bash
python scripts/train_ppo.py --checkpoint checkpoints/ppo.zip --seed 0
python scripts/ppo_routes.py infer --checkpoint checkpoints/ppo.zip --start H01 --end H02 --departure 08:00 --budget-min 240
python scripts/ppo_routes.py evaluate --checkpoint checkpoints/ppo.zip --iterations 100 --output reports/ppo-evaluation.json
```

Training uses 60–240 minute budgets. Use `python scripts/train_ppo.py --help`
for rollout, model and optimisation settings. Both training commands create
`checkpoints/ppo.zip` by default; `--checkpoint` changes the output path.
Evaluation uses the fixed held-out
scenarios saved by the training configuration and reports failures for each
method; successful score and travel-time averages exclude failed cases. The
checkpoint is tied to the exact matrix file by SHA-256. SBX collects complete
rollouts, so training may exceed the requested step count by up to one rollout
minus one step. The older custom PyTorch `.pt` checkpoints are not compatible.
`PPOSearch` in `busroute.routing.ppo_od` implements the same callable solver
interface as `ALNS` and `VNS`, using a loaded policy and its matrix features.

The simulator uses each leg's departure slot, with no waiting or POI dwell time.
The mask's return-time estimate freezes the arrival slot, so a slot change can
still leave the agent without a valid continuation. Matrix feasibility itself is
an approximation and should not be interpreted as field-validated road legality.
