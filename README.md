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
python scripts/train_ppo.py     --graph data/processed                     # Routing (PPO + GNN)
pytest                                                                     # Tests
```

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

In Python, load `ODMatrix.from_csv(path)` and call `alns(...)` or `vns(...)`
from `busroute.routing.metaheuristics`. Both return a typed `RouteResult`.
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
