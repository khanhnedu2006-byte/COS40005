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
python scripts/ppo_routes.py train --checkpoint checkpoints/ppo.zip         # OD-matrix PPO
pytest                                                                     # Tests
```

`build_graph.py` generates `nodes.parquet` and `edges.parquet` in its output
directory. The older `scripts/train_ppo.py` only loads those graph files; its
graph-based training is not implemented. The PPO command above instead uses
the OD matrix in `data/final/od_matrix.csv`.

## Conventions
- Variable / function / file names in **English**.
- All bus restrictions are **hard-coded** in `src/busroute/config/` – OSM is used only as the base map.
- Never commit raw GPS logs or any driver video (see `.gitignore`).

## Orienteering with ALNS and VNS

The OD-matrix solvers maximise the scores of unique POIs between a fixed start
and end, subject to a travel-time budget. Run from the repository root after
installing the package:

```bash
python scripts/solve_orienteering.py --algorithm alns --start H01 --end H02 --departure 08:00 --budget-min 240 --iterations 100 --seed 42
python scripts/solve_orienteering.py --algorithm vns --start H01 --end H02 --departure 08:00 --budget-min 240 --iterations 100 --seed 42
```

Use `--matrix data/final/od_matrix_mock.csv` for a mock-data smoke test. Output is
JSON with the selected stops, total score, travel minutes and collected POI IDs.
The mock matrix is not suitable for reporting real results.

In Python, construct `ALNS` or `VNS` and call the instance. Both implement the
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
pipeline reads `data/final/od_matrix.csv` by default; pass `--matrix PATH` before
the subcommand to use another matrix. For example:

```bash
python scripts/ppo_routes.py train --checkpoint checkpoints/ppo.zip --seed 0
python scripts/ppo_routes.py infer --checkpoint checkpoints/ppo.zip --start H01 --end H02 --departure 08:00 --budget-min 240
python scripts/ppo_routes.py evaluate --checkpoint checkpoints/ppo.zip --iterations 100 --output reports/ppo-evaluation.json
```

Training defaults to 200,000 steps and 60–240 minute budgets. Use `train --help`
for rollout, model and optimisation settings. Evaluation uses the fixed held-out
scenarios saved by the training configuration and reports failures for each
method; successful score and travel-time averages exclude failed cases. The
checkpoint is tied to the exact matrix file by SHA-256. SBX collects complete
rollouts, so training may exceed the requested step count by up to one rollout
minus one step. The older custom PyTorch `.pt` checkpoints are not compatible.

The simulator uses each leg's departure slot, with no waiting or POI dwell time.
The mask's return-time estimate freezes the arrival slot, so a slot change can
still leave the agent without a valid continuation. Matrix feasibility itself is
an approximation and should not be interpreted as field-validated road legality.
