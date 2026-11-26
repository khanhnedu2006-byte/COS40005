# Web application and development guide

## Setup and launch

Run from the repository root after installing optional web and test dependencies:

```bash
uv sync --extra web --extra test
uv run --extra web python -m Web
```

This opens the local MapLibre map at `http://127.0.0.1:8000`. Use `--port 8001 --no-browser`
to choose another port without opening a browser. `python Web/serve_orienteering.py`
is also supported after installing the project; the canonical command is
`python -m Web`. Map tiles and labels require internet access; solver data is local.

The existing text solver and desktop viewer keep their separate entry points:

```bash
python scripts/solve_orienteering.py --algorithm dijkstra --start H01 --end H02 --budget-min 240
uv run --extra visualization python scripts/visualize_orienteering.py --algorithm alns --start H01 --end H02 --budget-min 240
```

## Road search

Choose Dijkstra, ALNS, VNS, ACO, or reduced-graph PPO with distinct configured
endpoints, a minute budget, and a departure time. Defaults are 240 minutes,
08:00, 100 iterations, and seed 0. Dijkstra compares candidate connections by
estimated travel time. ALNS, VNS, and ACO maximize unique POI score within that
time, preferring lower narrow-road-weighted distance when scores tie. Known
widths below 5.5 m receive 2× cost for this tie-break; unknown widths remain
unpenalized.

The PPO solver uses a saved SB3 `MaskablePPO` ZIP checkpoint (defaulting to
`checkpoints/ppo_reduced.zip`) for deterministic edge-by-edge inference on the
reduced stage1 road graph (with `via_nodes` geometry reconstruction and
road-width awareness). See the [PPO guide](ppo.md).

Each isolated worker loads the complete base graph once. Dijkstra, ALNS, VNS,
and ACO build slot-specific directed Dijkstra trees for the selected sources.
POI connections, their edge IDs, and passed-POI masks are cached for search
trials. A POI detour `A → P → B` uses two connections even if the direct
`A → B` connection skips `P`. The destination cannot be crossed before the
final leg. The solver caches one path per point pair and traffic slot and
compares their actual replay times when a trip crosses slots.

Inputs are `data/stage1/nodes_base.parquet`, `edges_base.parquet`,
`data/final/duong_pho_clean.csv`, and `configs/pois.py` / `hubs.py`. Points
beyond the 300 m road-node snapping limit stay on the map with an explanation
but are disabled as endpoints and excluded as inserted waypoints. POIs earn
score once within 100 m of the approximate road geometry; selecting a waypoint
does not automatically award its score.

## Process and HTTP contracts

Only one job runs at a time. Cancel sets a cooperative flag, with bounded
process termination as fallback. A completed result is preserved if Cancel
arrives later. Only a final outcome is retained; no search history or animation
is recorded.

- `GET /api/bootstrap`: configured markers, snapping availability, defaults, and status.
- `GET /api/status`: road-data readiness, failure reason, and startup timing.
- `POST /api/runs`: `algorithm`, `start`, `end`, `budget_min`, `departure`,
  `iterations`, `seed`, optional `checkpoint`. Solver keys are `dijkstra`,
  `alns`, `vns`, `aco`, and `ppo`. The PPO solver requires `checkpoint`.
  Kilometre-budget fields are rejected. Invalid inputs give 422, overlapping
  runs 409, unavailable road data 503.
- `GET /api/runs/{id}`: `{status, result, message}`. Terminal states are
  `result`, `error`, and `stopped`; running/error/stopped results are null.
- `POST /api/runs/{id}/stop`: cancel or return the already-completed status.
- `GET /api/runs/{id}/result`: download the complete JSON result after success.

The browser polls every 350 ms, with 1.5-second retries after network failure.
Replacing a job makes its old run URLs return 404. Python contracts are defined
in `Web/types.py`; corresponding browser contracts are in
`Web/assets/contracts.js`. The legacy OD-matrix CLI/native-viewer interfaces
remain independent.

## Completed route

The final snapshot includes selected waypoint IDs, `duration_min`, `distance_km`,
unique score, collected POIs, `LineString` geometry, known-narrow/unknown-width
lengths, and road instructions. Geometry, metrics, and directions all derive
from the same cached edge sequence. Numbered waypoint buttons and road
directions focus the map. Passed POIs are highlighted in gold; their names
appear in the guide.

Road shapes are approximated by lines between graph nodes because the base
Parquet tables omit edge curves (except for PPO on the reduced graph, which
expands `via_nodes` back through base-node coordinates). Turn instructions use
those chords and are explicitly labeled as estimates. Travel time uses the
historical road-class × slot coefficients calibrated to TomTom. Each edge uses
its own departure slot, shared by all road solvers. Live traffic, dwell times,
and legal bus restrictions are not modeled. The central-ring overlay supplies
context only.

Road instructions appear above the waypoint list. Each instruction retains its
exact section of the approximate route geometry: selecting it highlights that
section in gold and fits it on the map. Clicking the blue route selects the
corresponding instruction and opens its road name. Unnamed roads stay explicit.

## Annotation and documentation conventions

Python files in `Web/` annotate function parameters, return values, and mutable
state whose initial value would otherwise hide its eventual type. `TypedDict`
contracts describe the JSON boundary; no runtime solver class is sent over HTTP.
Multiprocessing queues and cancellation events have explicit types.

Use Google-style docstrings: a concise summary followed by `Args`, `Returns`,
`Raises`, `Attributes`, or `Yields` where applicable. Document units and explain
missing-value semantics. Methods returning `None` need not repeat a `Returns`
section. Explain terminal status values and nullable result/error fields.

Browser code remains plain JavaScript. Its JSDoc documents payloads, state,
parameters, and asynchronous results; `Web/assets/contracts.js` is type
documentation only. HTML and CSS remain standard browser assets. Vendored
MapLibre files are left unmodified and retain their upstream license.

Static checks, using the project's interpreter and the separately installed
`ty` and `ruff` tools:

```bash
ty check Web --python .venv
ruff check Web scripts/visualize_orienteering.py src/busroute/routing/metaheuristics.py src/busroute/routing/od_evaluation.py
```

The requested capitalized package name `Web` is explicitly exempted from the
module-name lint rule in its package entry files. Other checks still apply.
When Node.js is available for development, `node --check Web/assets/app.js`
checks JavaScript syntax; Node.js is not needed to run the app.

## Regression tests

```bash
uv run --extra web --extra test python -m pytest tests/test_orienteering.py tests/test_visualize_orienteering.py tests/test_web_orienteering.py tests/test_ppo_env.py -q
```

The checks cover deterministic results, feasible snapshots, zero iterations,
input validation, stop/repeat behavior, final-only queue payloads, geometry
curves/direction, cache reuse, loading/failure states, typed OpenAPI responses,
packaged static-resource availability, and reduced-graph PPO environment/checkpoint
integration. Some restricted sandboxes block the local IPC used by `TestClient`
and spawned workers; use a normal local terminal for these integration tests.

## Browser verification

Exercise all five solvers (`dijkstra`, `alns`, `vns`, `aco`, `ppo`) with the same
real-data endpoints, departure, and budget. Verify that only completed routes
appear and that the guide's stop order, travel time, and score match the
downloaded JSON. Check numbered markers, Previous/Next, Fit route, Edit search,
cancellation, and reruns. Narrow screens must keep the stop list scrollable and
search settings accessible. Verify that a cancelled or failed run does not
display an old itinerary.

Also check failed tiles, missing geometry, interrupted status requests, and
viewport preservation when results arrive. Missing legs must stay visibly
missing. Confirm graph-load count stays at one across runs and cached legs are
reused. Measure solver and geometry time separately; test API and map
responsiveness during a long search. No SSE requests, intermediate route
layers, playback controls, or continuous rendering loop should remain.

## Known limits

The application uses configured locations (points beyond the 300 m snapping
limit are disabled). It does not geocode arbitrary pins, compute live traffic,
model dwell times, simulate vehicle movement, or prove exact optimality or legal
bus-road compliance. Base-graph geometry uses node-to-node chords and road-width
matching remains partial. The current run is held in memory, and map tiles
require network access. Browser map rendering also depends on WebGL and device
graphics.
