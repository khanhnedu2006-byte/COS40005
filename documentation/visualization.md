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

The desktop viewer uses `src/busroute/viz/road_paths.py` for coordinate loading
and road geometry. The restored browser application uses `Web/road_paths.py`;
the desktop viewer does not require the browser's FastAPI dependencies.

## Browser map with MapLibre and FastAPI

The current application is grouped in the root-level `Web/` package. MapLibre
renders an OpenFreeMap basemap, while Python evaluates routes and reconstructs
their road geometry. Users choose endpoints and solver inputs, then receive only
the final path. The route guide shows names and numbered stops, total travel
distance, score and map-linked stop selection. There is no web process animation.

The map uses zoom-aware labels and marker popups. It preserves the user's
viewport when a result arrives; Fit route changes the camera explicitly.

Performance comes from sparse graph searches, cached directed POI connections,
and an isolated solver process per run. Only the final expanded road route is
sent to the browser; graph searches are not repeated for every insertion trial.
The map uses tiles rather than loading the full road graph into JavaScript.

The browser is the most complete approach here, but it requires a local server,
WebGL support and internet access for map tiles/labels. The JavaScript and CSS
are bundled locally; no Node.js build or map-service account is required.

## Interpretation and limits

Blue denotes the final route and its numbered stops. The selected itinerary
stop receives a larger marker. The central-ring boundary supplies context.

The browser now evaluates graph-derived estimated travel times, physical distances and unique POI scores.
ALNS/VNS add waypoint detours within a minute budget. The guide also lists road
names, distances and estimated turns. Passed POIs within 100 m of the approximate
path are highlighted in gold. Base tables lack road curves, so graph-node chords
approximate geometry. Width matching remains partial; travel times use calibrated class/slot coefficients; live traffic and legal bus
restrictions are not modeled. Legacy CLI/native-viewer routes still use the OD
matrix and its time-based evaluator.

External references: [OSMnx routing](https://osmnx.readthedocs.io/en/stable/user-reference.html#module-osmnx.routing),
[MapLibre examples](https://maplibre.org/maplibre-gl-js/docs/examples/), and
[OpenFreeMap](https://openfreemap.org/).
