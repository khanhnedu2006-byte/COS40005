# Visualization approaches

## Static GeoJSON and route maps with Folium

The first approach places `data/references/central_ring.geojson` on a tiled map.
The polygon supplies geographic context for the central restriction boundary.
The existing `src/busroute/viz/route_map.py` can also save a supplied graph-node
route and POIs as an HTML map.

Folium is useful for inspecting a boundary or sharing a final route. Its saved
HTML is not a live view of a running solver. The static route helper connects
the coordinates it receives and does not expose search operators or a timeline.

## Earlier native OSMnx and Matplotlib viewer

`scripts/visualize_orienteering.py` reuses the solver CLI's arguments and opens a
Tk/Matplotlib window. It draws OSM roads, the central ring, stops, current and
candidate routes, and best-route progress. Pause/resume and speed controls make
operator events inspectable without a browser application.

This approach keeps the interface in Python and is useful for desktop
demonstrations. Loading a large graph and repeatedly redrawing a figure can
produce noticeable startup cost and lag. Dense point annotations also make
the plotted map harder to read than a zoom-aware tiled map.

The desktop viewer now imports its shared coordinate/geometry helper from
`Web/road_paths.py`; the browser's FastAPI dependencies are not imported merely
by using that helper.

## Browser map with MapLibre and FastAPI

The current application is grouped in the root-level `Web/` package. MapLibre
renders an OpenFreeMap basemap, while Python evaluates routes and reconstructs
their road geometry. Users choose endpoints and solver inputs, then receive only
the final path. The route guide shows names and numbered stops, total travel
time, score and map-linked stop selection. There is no web process animation.

The map uses zoom-aware labels and marker popups. It preserves the user's
viewport when a result arrives; Fit route changes the camera explicitly.

Performance comes from separating responsibilities:

- The basemap uses tiles, never the complete OSM graph.
- The graph initializes once in a background thread.
- Each solver runs in an isolated process.
- Only a terminal result is retained and polled; no operator history is built.
- Final-route geometry is fetched separately in bounded batches.
- Server and browser caches reuse directed legs across runs.
- Map sources update in place without a continuous animation loop.

The browser is the most complete approach here, but it requires a local server,
WebGL support and internet access for map tiles/labels. The JavaScript and CSS
are bundled locally; no Node.js build or map-service account is required.

## Interpretation and limits

Blue denotes the final route and its numbered stops. The selected itinerary
stop receives a larger marker. The central-ring boundary supplies context.

The OD matrix owns scores, times and feasibility. OSM reconstructs approximate
road shapes by distance; those shapes can differ from the provider routes used
to build the matrix. Missing paths and excessive snap distances are reported,
without drawing straight-line substitutes. A visible route is not certification
that a bus meets every road or time-dependent restriction.

External references: [OSMnx routing](https://osmnx.readthedocs.io/en/stable/user-reference.html#module-osmnx.routing),
[MapLibre examples](https://maplibre.org/maplibre-gl-js/docs/examples/), and
[OpenFreeMap](https://openfreemap.org/).
