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
pytest                                                                     # Tests
```

`build_graph.py` generates road graph files in its output directory. PPO uses
the OD matrix in `data/final/od_matrix.csv`, rather than those graph files.

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
uv sync --extra visualization
uv run --extra visualization python scripts/visualize_orienteering.py --algorithm alns --start H01 --end H02 --departure 08:00 --budget-min 240 --iterations 100 --seed 42
```

Alternatively, install plotting support with `pip install -e '.[visualization]'`.
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

For a street map with location inputs and a final-route guide, run:

```bash
uv run --extra web python -m Web
```

This starts a local server at `http://127.0.0.1:8000` and opens your browser.
Alternatively, use `pip install -e '.[web]'` followed by
`python -m Web`. No Node.js, frontend build, map account
or API key is required. MapLibre GL JS 5.24.0 is bundled locally; the OpenFreeMap
basemap, labels and glyphs require internet access and a WebGL-capable browser.

Choose existing hubs/POIs by searching their IDs or names, or click a map marker
to set an endpoint. Choose ALNS, VNS, Ant Colony (ACO), or Dijkstra; set departure
time and travel-time budget, then Find route. The heuristic solvers also use
iterations and random seed; Dijkstra disables those inputs. Computation runs in
a separate process, and Cancel stops an active search. Only the completed route
is displayed; intermediate search states are not recorded or animated by the web
app. Existing command-line solver commands remain available.

The final route opens immediately in a map-linked itinerary with start and
destination names, travel time, POI score and numbered stops. Click a stop or use
Previous/Next to locate it on the map. Fit route adjusts the viewport explicitly;
completion preserves your pan/zoom. Edit search returns to the inputs, and
Download JSON saves the final matrix-based result.

The OSM graph loads once in the background. The API and browser cache directed
leg geometry across runs. Blue marks the final road path and numbered locations;
the central-ring boundary provides context. Reconstructed paths may differ from
the matrix provider's routes; missing road connections are reported without
substitute lines. Stop order and totals remain available while roads initialize
or if geometry fails.

Launch options: `--matrix data/final/od_matrix_mock.csv`, `--port 8001`, and
`--no-browser`. The server holds the current run in memory until replaced or
restarted; only one search runs at a time. Ctrl+C stops the server and worker.
If roads fail to initialize, check the server's OSM-cache/network message and
restart. If tiles fail, check internet access and use Retry map. Automated web
API tests require `uv sync --extra web --extra test` before running pytest.

PPO learns by simulation for the requested route when no checkpoint is supplied;
`--ppo-steps` controls its learning budget. `--iterations` and `--seed`
configure ALNS, VNS and ant colony optimisation (ACO), while checkpoint-based PPO uses its trained
policy for inference. The Dijkstra-style baseline seeks a short feasible route,
without trying to maximise POI score; it uses time-dependent arrival-minute
labels and does not model waiting, so it is not guaranteed to find the exact
shortest route when availability or travel times change between slots.
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

Install the project dependencies with `pip install -e .`. PPO uses SBX's JAX
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
