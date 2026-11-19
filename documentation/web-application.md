# Web application

Run `python -m Web` from an editable checkout to open the local MapLibre map.
Use `--port 8001 --no-browser` to choose another port without opening a browser.
Map tiles and labels require internet access; solver data is local.

## Road search

Choose Dijkstra, ALNS, VNS, ACO, edge PPO or Waypoint PPO with distinct configured endpoints and a minute
budget and departure time. Defaults are 240 minutes, 08:00, 100 iterations and seed 0. Dijkstra compares candidate connections by estimated travel time. ALNS, VNS and ACO maximize unique POI score within that time, preferring
lower narrow-road-weighted distance when scores tie. Known widths below 5.5 m
receive 2× cost for this tie-break; unknown widths remain unpenalized.
Both PPO solvers use saved SB3 ZIP checkpoints for deterministic inference.
Edge PPO defaults to `checkpoints/ppo_road.zip`. Waypoint PPO uses
`checkpoints/ppo_waypoint.zip`, with Dijkstra legs connecting its chosen POIs
and a verified continuation to the destination. See the
[edge PPO training guide](ppo-road.md) and [Waypoint PPO commands](algorithms.md#waypoint-ppo).

Each isolated worker loads the complete base graph once. Dijkstra, ALNS, VNS and
ACO build slot-specific directed Dijkstra trees for the selected sources. POI connections, their edge IDs and
passed-POI masks are cached for search trials. A POI detour A → P → B uses two
connections even if the direct A → B connection skips P. The destination cannot
be crossed before the final leg. The solver caches one path per point pair and
traffic slot and compares their actual replay times when a trip crosses slots.

Inputs are `data/stage1/nodes_base.parquet`, `edges_base.parquet`,
`data/final/duong_pho_clean.csv`, and `configs/pois.py` / `hubs.py`.
Points beyond the 300 m road-node snapping limit stay on the map with an
explanation but are disabled as endpoints and excluded as inserted waypoints.
POIs earn score once within 100 m of the approximate road geometry; selecting a
waypoint does not automatically award its score.

## Process and HTTP contracts

Only one job runs at a time. Cancel sets a cooperative flag, with bounded process
termination as fallback. A completed result is preserved if Cancel arrives later.
Only a final outcome is retained; no search history or animation is recorded.

- `GET /api/bootstrap`: configured markers, snapping availability, defaults and status.
- `GET /api/status`: road-data readiness, failure reason and startup timing.
- `POST /api/runs`: `algorithm`, `start`, `end`, `budget_min`, `departure`, `iterations`, `seed`, optional `checkpoint`.
  Solver keys are `dijkstra`, `alns`, `vns`, `aco`, `ppo` and `ppo_waypoint`.
  Both PPO solvers require `checkpoint`.
  Kilometre-budget fields are rejected.
  Invalid inputs give 422, overlapping runs 409, unavailable road data 503.
- `GET /api/runs/{id}`: `{status, result, message}`. Terminal states are `result`,
  `error` and `stopped`; running/error/stopped results are null.
- `POST /api/runs/{id}/stop`: cancel or return the already-completed status.
- `GET /api/runs/{id}/result`: download the complete JSON result after success.

The browser polls every 350 ms, with 1.5-second retries after network failure.
Replacing a job makes its old run URLs return 404. Python contracts are defined
in `Web/types.py`; corresponding browser contracts are in `assets/contracts.js`.
The old OD-matrix CLI/native-viewer interfaces remain independent.

## Completed route

The final snapshot includes selected waypoint IDs, `duration_min`, `distance_km`, unique score,
collected POIs, LineString geometry, known-narrow/unknown-width lengths and road
instructions. Geometry, metrics and directions all derive from the same cached
edge sequence. Numbered waypoint buttons and road directions focus the map.
Passed POIs are highlighted in gold; their names appear in the guide.

Road shapes are approximated by lines between graph nodes because the base
Parquet tables omit edge curves. Turn instructions use those chords and are
explicitly labeled as estimates. Travel time uses the historical road-class × slot coefficients calibrated to TomTom. Each edge uses its own departure slot, sharing the clock between PPO and the other road solvers. Live traffic, dwell times and legal bus restrictions are not modeled. The central-ring overlay supplies context only.

Road instructions appear above the waypoint list. Each instruction retains its
exact section of the approximate route geometry: selecting it highlights that
section in gold and fits it on the map. Clicking the blue route selects the
corresponding instruction and opens its road name. Unnamed roads stay explicit.
