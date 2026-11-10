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
python scripts/stage1_build_base.py                                        # OSM base map → data/stage1, data/cache/road_graph_drive.graphml
python scripts/fetch_routes.py fetch && python scripts/fetch_routes.py build   # TomTom durations → data/final/od_matrix.csv, test_legs.csv (needs .env key)
python scripts/calibrate_estimator.py                                      # Fit speed per road group × slot → configs/estimator_params.json
python scripts/build_road_graph.py                                         # RL dataset → data/final/{pois,road_nodes,road_edges}.csv
python scripts/check_road_graph.py                                         # Validate against TomTom → reports/road_graph_validation.csv
pytest                                                                     # Tests
```
RL dataset description: [`data/final/README.md`](data/final/README.md), details in [`data/final/road_graph_card.md`](data/final/road_graph_card.md).
GPS logs (`scripts/normalize_dataset.py`, `scripts/run_clean_gps.py`, `scripts/build_gps_trips.py`) are only used to check TomTom times.

## Conventions
- Variable / function / file names in **English**.
- Central-ring ban is hard-coded in `src/busroute/config/restrictions.py` + `data/references/central_ring.geojson`; road width from `data/references/danang_duong_pho.csv` (source: Da Nang City portal, danang.gov.vn).
- Never commit raw GPS logs or any driver video (see `.gitignore`).
