# Development, documentation style and verification

## Setup and commands

Run from the repository root after installing optional web/test dependencies:

```bash
uv sync --extra web --extra test
uv run --extra web python -m Web
```

For a browser-free startup check, append `--no-browser`. Choose a different port
with `--port 8001`. `python Web/serve_orienteering.py` is also supported after
installing the project; the canonical command is `python -m Web`.

The existing text solver and desktop viewer keep their separate entry points:

```bash
python scripts/solve_orienteering.py --algorithm dijkstra --start H01 --end H02 --budget-min 240
uv run --extra visualization python scripts/visualize_orienteering.py --algorithm alns --start H01 --end H02 --budget-min 240
```

## Annotation and documentation conventions

Python files in `Web/` annotate function parameters, return values and mutable
state whose initial value would otherwise hide its eventual type. TypedDict
contracts describe the JSON boundary; no runtime solver class is sent over HTTP.
Multiprocessing queues and cancellation events have explicit types.

Use Google-style docstrings: a concise summary followed by `Args`, `Returns`,
`Raises`, `Attributes` or `Yields` where applicable. Document units and explain
missing-value semantics. Methods returning None need not repeat a `Returns`
section. Explain terminal status values and nullable result/error fields.

Browser code remains plain JavaScript. Its JSDoc documents payloads, state,
parameters and asynchronous results; `contracts.js` is type documentation only.
HTML and CSS remain standard browser assets. Vendored MapLibre files are left
unmodified and retain their upstream license.

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
uv run --extra web --extra test python -m pytest tests/test_orienteering.py tests/test_visualize_orienteering.py tests/test_web_orienteering.py -q
```

The checks cover deterministic results, feasible snapshots, zero iterations,
input validation, stop/repeat behavior, final-only queue payloads, geometry
curves/direction, cache reuse, loading/failure states, typed OpenAPI responses,
and packaged static-resource availability. Some restricted sandboxes block the
local IPC used by TestClient and spawned workers; use a normal local terminal
for these integration tests.

PPO has its own environment and training tests. Training tests can require more
time and machine resources; they are separate from browser visualization checks.

## Browser verification

Exercise all four algorithms with the same real-data endpoints, departure and
budget. Verify that only completed routes appear and that the guide's stop order,
travel time and score match the downloaded JSON. Check numbered markers,
Previous/Next, Fit route, Edit search, cancellation and reruns. Narrow screens
must keep the stop list scrollable and search settings accessible. Verify that
a cancelled or failed run does not display an old itinerary.

Also check failed tiles, missing geometry, interrupted status requests and
viewport preservation when results arrive. Missing legs must stay visibly
missing. Confirm graph-load count stays at one across runs and cached legs are
reused. Measure solver and geometry time separately; test API and map
responsiveness during a long search. No SSE requests, intermediate route
layers, playback controls or continuous rendering loop should remain.

## Known limits

The application uses the selected matrix's existing locations. It does not
geocode arbitrary pins, compute live traffic, simulate vehicle movement, or
prove exact optimality or legal bus-road compliance. The matrix and displayed
OSM paths can differ. The current run is held in memory, and map tiles require
network access. Browser map rendering also depends on WebGL and device graphics.
