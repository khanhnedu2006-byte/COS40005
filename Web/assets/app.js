/* MapLibre renders the map; operator playback never rebuilds it. */
"use strict";
/**
 * @typedef {import("./contracts.js").AlgorithmName} AlgorithmName
 * @typedef {import("./contracts.js").SearchParameters} SearchParameters
 * @typedef {import("./contracts.js").RouteSnapshot} RouteSnapshot
 * @typedef {import("./contracts.js").PointMetadata} PointMetadata
 * @typedef {import("./contracts.js").SearchEvent} SearchEvent
 * @typedef {import("./contracts.js").PlaybackFrame} PlaybackFrame
 * @typedef {import("./contracts.js").GeometryLeg} GeometryLeg
 * @typedef {import("./contracts.js").GraphStatusPayload} GraphStatusPayload
 * @typedef {import("./contracts.js").BootstrapPayload} BootstrapPayload
 * @typedef {import("./contracts.js").BrowserState} BrowserState
 */

(() => {
  /** Resolve a shell control by ID.
   * @param {string} id - Stable element ID from index.html.
   * @returns {HTMLElement | SVGElement | null}
   */
  const $ = (id) => document.getElementById(id);
  /** Create an empty GeoJSON source value.
   * @returns {{type: "FeatureCollection", features: Object[]}}
   */
  const EMPTY = () => ({ type: "FeatureCollection", features: [] });
  const STYLE = "https://tiles.openfreemap.org/styles/bright";
  const terminal = new Set(["result", "error", "stopped"]);
  /** @type {BrowserState} */
  const state = {
    points: [],
    byId: new Map(),
    frames: [],
    index: -1,
    playing: true,
    runId: null,
    active: false,
    stream: null,
    ready: false,
    graphReady: false,
    geometryCache: new Map(),
    pending: new Map(),
    renderToken: 0,
    busy: false,
    generation: 0,
    params: null,
    lastAdvance: 0,
    mapReady: false,
    result: null,
    renderedRoutes: {},
    markerSelection: null,
    lastAffected: null,
  };
  let map;
  let controlsScheduled = false;

  /** Batch control updates to one browser animation frame.
   * @returns {void}
   */
  function scheduleControls() {
    if (controlsScheduled) return;
    controlsScheduled = true;
    requestAnimationFrame(() => {
      controlsScheduled = false;
      updateControls();
    });
  }

  /** Show or clear a form-level validation/server error.
   * @param {string} message - User-facing text; empty text clears the error.
   * @returns {void}
   */
  function formError(message) {
    $("form-error").textContent = message;
    $("form-error").hidden = !message;
  }
  /** Show or clear the basemap failure notice.
   * @param {string} message - User-facing network or rendering explanation.
   * @returns {void}
   */
  function mapNotice(message) {
    $("map-notice-text").textContent = message;
    $("map-notice").hidden = !message;
  }
  /** Resolve the current endpoint input to matrix metadata.
   * @param {"start" | "end"} input - Endpoint input ID.
   * @returns {PointMetadata | undefined}
   */
  function selectedPoint(input) {
    return state.byId.get($(input).value.trim());
  }
  /** Validate endpoint IDs, update names, and refresh selected markers.
   * @returns {void}
   */
  function endpointNames() {
    for (const name of ["start", "end"]) {
      const point = selectedPoint(name);
      $(name + "-name").textContent = point
        ? point.name
        : "Choose a location from the list";
      $(name).setCustomValidity(
        point ? "" : "Choose a point from the selected matrix",
      );
    }
    updateMarkers();
  }
  /** Toggle the input panel or explicitly set its collapsed state.
   * @param {boolean} [force] - Requested state; omit to toggle.
   * @returns {void}
   */
  function collapse(force) {
    const panel = document.querySelector(".panel");
    const collapsed =
      force === undefined ? !panel.classList.contains("collapsed") : force;
    panel.classList.toggle("collapsed", collapsed);
    $("settings-body").hidden = collapsed;
    $("collapse").textContent = collapsed ? "+" : "−";
    $("collapse").setAttribute("aria-expanded", String(!collapsed));
    $("collapse").setAttribute(
      "aria-label",
      collapsed ? "Expand settings" : "Collapse settings",
    );
  }
  $("collapse").onclick = () => collapse();
  $("start").oninput = endpointNames;
  $("end").oninput = endpointNames;

  /** Update controls to reflect the selected solver's actual objective.
   * @returns {void}
   */
  function algorithmInputs() {
    const shortest = $("algorithm").value === "dijkstra";
    $("iterations").disabled = shortest;
    $("seed").disabled = shortest;
    $("algorithm-objective").textContent = shortest
      ? "Minimise travel time. Iterations and random seed do not apply."
      : "Maximise unique POI score within the time budget; shorter time breaks ties.";
  }
  $("algorithm").onchange = algorithmInputs;

  /** Fetch a JSON API response and turn HTTP failures into readable errors.
   * @template T
   * @param {string} path - Same-origin API path.
   * @param {RequestInit} [options={}] - Fetch method, headers and body.
   * @returns {Promise<T>}
   */
  async function api(path, options = {}) {
    const response = await fetch(path, options);
    let body;
    try {
      body = await response.json();
    } catch {
      throw new Error("The server returned an unreadable response");
    }
    if (!response.ok) {
      const detail = Array.isArray(body.detail)
        ? body.detail.map((d) => d.msg).join("; ")
        : body.detail;
      const error = new Error(detail || `Request failed (${response.status})`);
      error.status = response.status;
      throw error;
    }
    return body;
  }
  /** Encode a JSON POST without introducing a browser HTTP library.
   * @param {unknown} body - Serializable request body.
   * @returns {RequestInit}
   */
  const post = (body) => ({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  /** Update endpoint colors and affected stops without rebuilding labels.
   * @param {string[]} [affected=[]] - IDs highlighted for the displayed operator.
   * @returns {void}
   */
  function updateMarkers(affected = []) {
    if (!state.mapReady) return;
    const features = state.points.map((p) => ({
      type: "Feature",
      id: p.id,
      geometry: { type: "Point", coordinates: [p.lon, p.lat] },
      properties: {
        id: p.id,
        name: p.name,
        label: `${p.id}  ${p.name}`,
        kind: p.kind,
        endpoint:
          p.id === $("start").value
            ? "start"
            : p.id === $("end").value
              ? "end"
              : "none",
      },
    }));
    const selection = JSON.stringify([
      $("start").value,
      $("end").value,
      state.points.length,
    ]);
    if (state.markerSelection !== selection) {
      map.getSource("points").setData({ type: "FeatureCollection", features });
      state.markerSelection = selection;
    }
    const highlight = JSON.stringify(affected);
    if (state.lastAffected !== highlight) {
      map.getSource("affected").setData({
        type: "FeatureCollection",
        features: features.filter((f) => affected.includes(f.id)),
      });
      state.lastAffected = highlight;
    }
  }

  /** Install boundary, route, marker and label layers after a style load.
   * @returns {void}
   */
  function installLayers() {
    if (map.getSource("points")) return;
    map.addSource("ring", { type: "geojson", data: state.ring || EMPTY() });
    map.addLayer({
      id: "ring-fill",
      type: "fill",
      source: "ring",
      paint: { "fill-color": "#bb6b35", "fill-opacity": 0.07 },
    });
    map.addLayer({
      id: "ring-line",
      type: "line",
      source: "ring",
      paint: {
        "line-color": "#bb6b35",
        "line-width": 1.5,
        "line-dasharray": [4, 3],
      },
    });
    for (const [name, color, width, opacity] of [
      ["best", "#3d5fd9", 9, 0.4],
      ["current", "#176b5b", 4, 0.95],
      ["candidate", "#e7a33e", 3, 1],
    ]) {
      map.addSource(name, { type: "geojson", data: EMPTY() });
      const paint = {
        "line-color": color,
        "line-width": width,
        "line-opacity": opacity,
      };
      if (name === "candidate") paint["line-dasharray"] = [2, 2];
      map.addLayer({
        id: name,
        type: "line",
        source: name,
        layout: { "line-cap": "round", "line-join": "round" },
        paint,
      });
    }
    map.addSource("points", { type: "geojson", data: EMPTY() });
    map.addSource("affected", { type: "geojson", data: EMPTY() });
    map.addLayer({
      id: "affected",
      type: "circle",
      source: "affected",
      paint: {
        "circle-radius": 14,
        "circle-color": "#e7a33e",
        "circle-opacity": 0.3,
        "circle-stroke-color": "#bc7b12",
        "circle-stroke-width": 2,
      },
    });
    map.addLayer({
      id: "points",
      type: "circle",
      source: "points",
      paint: {
        "circle-radius": ["case", ["!=", ["get", "endpoint"], "none"], 7, 5],
        "circle-color": [
          "match",
          ["get", "endpoint"],
          "start",
          "#176b5b",
          "end",
          "#3d5fd9",
          ["match", ["get", "kind"], "hub", "#16324f", "#6c7e8e"],
        ],
        "circle-stroke-color": "white",
        "circle-stroke-width": 2,
      },
    });
    map.addLayer({
      id: "point-labels",
      type: "symbol",
      source: "points",
      minzoom: 12.5,
      layout: {
        "text-field": ["step", ["zoom"], ["get", "id"], 14, ["get", "label"]],
        "text-font": ["Noto Sans Regular"],
        "text-size": 11,
        "text-offset": [0, 1.5],
        "text-anchor": "top",
        "text-max-width": 16,
      },
      paint: {
        "text-color": "#16324f",
        "text-halo-color": "#ffffff",
        "text-halo-width": 2,
      },
    });
    state.mapReady = true;
    state.renderedRoutes = {};
    state.markerSelection = null;
    state.lastAffected = null;
    updateMarkers();
    if (state.index >= 0) renderFrame(state.index);
  }

  /** Open a safe-text marker popup with endpoint-selection actions.
   * @param {PointMetadata} point - Selected location.
   * @param {[number, number]} coordinates - Longitude/latitude popup anchor.
   * @returns {void}
   */
  function showPoint(point, coordinates) {
    const content = document.createElement("div");
    content.className = "point-popup";
    const heading = document.createElement("h3");
    heading.textContent = point.name;
    const info = document.createElement("p");
    info.textContent = `${point.id} • ${point.kind === "hub" ? "Hub" : "Point of interest"} • Score ${point.score.toFixed(2)}`;
    content.append(heading, info);
    const popup = new maplibregl.Popup({ offset: 12 })
      .setLngLat(coordinates)
      .setDOMContent(content)
      .addTo(map);
    for (const [name, label] of [
      ["start", "Set start"],
      ["end", "Set destination"],
    ]) {
      const button = document.createElement("button");
      button.textContent = label;
      button.onclick = () => {
        $(name).value = point.id;
        endpointNames();
        popup.remove();
      };
      content.append(button);
    }
  }

  /** Create the MapLibre map and attach map-level interactions.
   * @returns {void}
   */
  function initializeMap() {
    if (!window.maplibregl) {
      mapNotice(
        "The map renderer could not load. Reload the page after checking the local server.",
      );
      return;
    }
    try {
      map = new maplibregl.Map({
        container: "map",
        style: STYLE,
        center: [108.225, 16.055],
        zoom: 11.5,
        attributionControl: true,
      });
      map.addControl(new maplibregl.NavigationControl(), "top-right");
      map.addControl(new maplibregl.ScaleControl(), "bottom-left");
      map.on("style.load", installLayers);
      map.on("error", (event) => {
        console.warn("Map data:", event.error?.message || "Unknown map error");
        mapNotice(
          "Some map data could not load. Check your internet connection, then retry the map.",
        );
      });
      map.on("click", "points", (event) => {
        const feature = event.features[0];
        const point = state.byId.get(feature.properties.id);
        if (point) showPoint(point, feature.geometry.coordinates);
      });
      map.on("mouseenter", "points", () => {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", "points", () => {
        map.getCanvas().style.cursor = "";
      });
      $("retry-map").onclick = () => {
        state.mapReady = false;
        mapNotice("");
        map.setStyle(STYLE);
      };
      window.setTimeout(() => {
        if (!state.mapReady)
          mapNotice(
            "The basemap is taking too long to load. Check your internet connection and retry.",
          );
      }, 15000);
    } catch (error) {
      mapNotice(
        `The map could not start: ${error.message}. Enable WebGL in your browser.`,
      );
    }
  }

  /** Fit the displayed route, or all points before a run, on explicit request.
   * @returns {void}
   */
  function fitRoute() {
    if (!map || !state.ready) return;
    const frame = state.frames[state.index];
    const route = frame?.candidate || frame?.current || frame?.best;
    const points = route
      ? route.route.map((id) => state.byId.get(id)).filter(Boolean)
      : state.points;
    if (!points.length) return;
    const bounds = new maplibregl.LngLatBounds();
    points.forEach((p) => bounds.extend([p.lon, p.lat]));
    const mobile = window.matchMedia("(max-width:700px)").matches;
    map.fitBounds(bounds, {
      padding: mobile
        ? { top: 180, bottom: 240, left: 35, right: 35 }
        : { top: 90, bottom: 220, left: 380, right: 350 },
      maxZoom: 15,
      duration: window.matchMedia("(prefers-reduced-motion:reduce)").matches
        ? 0
        : 500,
    });
  }
  $("fit").onclick = fitRoute;

  /** Poll graph readiness while keeping the map and form interactive.
   * @returns {Promise<void>}
   */
  async function refreshStatus() {
    try {
      /** @type {GraphStatusPayload} */
      const status = await api("/api/status");
      state.graphReady = status.graph === "ready";
      $("graph-status").textContent =
        status.graph === "ready"
          ? "Road geometry ready"
          : status.graph === "error"
            ? "Road geometry unavailable"
            : "Preparing road geometry…";
      $("graph-dot").className =
        "status-dot " +
        (status.graph === "ready"
          ? "ready"
          : status.graph === "error"
            ? "failed"
            : "");
      if (status.graph === "error")
        formError(
          `Road geometry could not load: ${status.message}. Restart the server after checking its OSM cache or connection.`,
        );
      if (status.graph === "loading") window.setTimeout(refreshStatus, 1500);
    } catch (error) {
      $("graph-status").textContent = "Local server disconnected";
      formError(error.message + ". Check that the local server is running.");
      window.setTimeout(refreshStatus, 3000);
    }
  }

  /** Combine a sparse event with the last frame and derive affected stops.
   * @param {SearchEvent} event - Next ordered server progress record.
   * @returns {PlaybackFrame}
   */
  function frameFrom(event) {
    const previous = state.frames.at(-1) || {};
    const frame = {
      best: previous.best,
      current: previous.current,
      candidate: previous.candidate,
      iteration: previous.iteration || 0,
      expanded: previous.expanded || 0,
      ...event,
      affected: [],
    };
    if (["initial_solution", "result", "search_end"].includes(event.event)) {
      frame.candidate = null;
      if (event.event !== "search_end")
        frame.current = event.current || event.best;
    }
    if (["insert", "remove", "replace", "relocate"].includes(event.action)) {
      const before = previous.candidate || previous.current || previous.best;
      const oldRoute = before?.route || [];
      const newRoute = event.candidate?.route || [];
      frame.affected = [
        ...new Set([
          ...oldRoute.filter((p) => !newRoute.includes(p)),
          ...newRoute.filter((p) => !oldRoute.includes(p)),
        ]),
      ];
      if (event.action === "relocate")
        frame.affected = newRoute.filter((p, i) => oldRoute[i] !== p);
    }
    return frame;
  }

  /** Reflect computation, playback, and download availability in controls.
   * @returns {void}
   */
  function updateControls() {
    $("run").disabled = !state.ready || state.active;
    $("stop").disabled = !state.active;
    $("timeline").disabled = !state.frames.length;
    $("timeline").max = Math.max(0, state.frames.length - 1);
    $("timeline").value = Math.max(0, state.index);
    $("frame-number").textContent =
      `${Math.max(0, state.index + 1)} / ${state.frames.length}`;
    $("play").disabled = !state.frames.length;
    $("step").disabled = state.index >= state.frames.length - 1;
    const atEnd = state.index >= state.frames.length - 1;
    const replay = atEnd && !state.active && state.frames.length > 0;
    $("play").textContent = replay
      ? "↺ Replay"
      : state.playing
        ? "Ⅱ Pause"
        : "▶ Play";
    $("play").setAttribute(
      "aria-label",
      replay
        ? "Replay search"
        : state.playing
          ? "Pause playback"
          : "Play playback",
    );
    $("playback-state").textContent = !state.frames.length
      ? "No steps yet"
      : !state.graphReady
        ? "Loading geometry"
        : state.busy
          ? "Preparing route"
          : atEnd && !state.active
            ? "Playback complete"
            : state.playing
              ? "Playing"
              : "Paused";
  }

  $("run-form").onsubmit = async (event) => {
    event.preventDefault();
    formError("");
    endpointNames();
    if (!$("run-form").reportValidity()) return;
    const algorithm = /** @type {AlgorithmName} */ ($("algorithm").value);
    /** @type {SearchParameters} */
    const params = {
      algorithm,
      start: $("start").value.trim(),
      end: $("end").value.trim(),
      departure: $("departure").value,
      budget_min: Number($("budget_min").value),
      iterations: algorithm === "dijkstra" ? 0 : Number($("iterations").value),
      seed: algorithm === "dijkstra" ? 0 : Number($("seed").value),
    };
    if (params.start === params.end) {
      formError("Choose different start and destination points.");
      return;
    }
    $("run").disabled = true;
    try {
      const response = await api("/api/runs", post(params));
      if (state.stream) state.stream.close();
      state.generation++;
      state.renderToken++;
      state.busy = false;
      state.renderedRoutes = {};
      state.lastAffected = null;
      state.frames = [];
      state.index = -1;
      state.params = params;
      state.runId = response.id;
      state.active = true;
      state.playing = true;
      state.result = null;
      $("inspector").hidden = true;
      $("download").removeAttribute("href");
      $("download").setAttribute("aria-disabled", "true");
      $("search-state").textContent =
        `${params.algorithm.toUpperCase()} is computing…`;
      $("chart-end").textContent = params.iterations;
      if (state.mapReady)
        for (const name of ["best", "current", "candidate", "affected"])
          map.getSource(name).setData(EMPTY());
      if (window.matchMedia("(max-width:700px)").matches) collapse(true);
      const runId = state.runId;
      const stream = new EventSource(`/api/runs/${runId}/events`);
      state.stream = stream;
      stream.onmessage = (message) => {
        if (state.runId !== runId) return;
        /** @type {SearchEvent} */
        const item = JSON.parse(message.data);
        if (item.sequence <= state.frames.length) return;
        state.frames.push(frameFrom(item));
        if (terminal.has(item.event)) {
          state.active = false;
          stream.close();
          if (item.event === "result") {
            state.result = item.best;
            $("search-state").textContent =
              "Search complete — playback available";
            $("download").href = `/api/runs/${runId}/result`;
            $("download").setAttribute("aria-disabled", "false");
          } else if (item.event === "error") {
            $("search-state").textContent = "Search failed";
            formError(item.message);
          } else
            $("search-state").textContent =
              "Search stopped — recorded steps retained";
        }
        scheduleControls();
      };
      stream.onopen = () => {
        if (state.active && state.runId === runId)
          $("search-state").textContent =
            `${params.algorithm.toUpperCase()} is computing…`;
      };
      stream.onerror = () => {
        if (state.active && state.runId === runId)
          $("search-state").textContent =
            "Connection interrupted — reconnecting…";
      };
    } catch (error) {
      formError(error.message);
    }
    updateControls();
  };
  $("stop").onclick = async () => {
    if (!state.active) return;
    $("stop").disabled = true;
    try {
      await api(`/api/runs/${state.runId}/stop`, { method: "POST" });
    } catch (error) {
      formError(error.message);
      updateControls();
    }
  };

  /** Collect unique directed leg requests across visible route roles.
   * @param {PlaybackFrame} frame - Snapshot whose road paths will be displayed.
   * @returns {Map<string, [string, string]>}
   */
  function legPairs(frame) {
    const pairs = new Map();
    for (const name of ["best", "current", "candidate"]) {
      const route = frame[name]?.route || [];
      for (let i = 1; i < route.length; i++) {
        const pair = [route[i - 1], route[i]];
        pairs.set(JSON.stringify(pair), pair);
      }
    }
    return pairs;
  }
  /** Fetch missing legs in bounded batches and reuse pending/cache entries.
   * @param {Map<string, [string, string]>} pairs - Directed ID pairs keyed by JSON.
   * @returns {Promise<void>}
   */
  async function ensureGeometry(pairs) {
    const missing = [...pairs].filter(
      ([key]) => !state.geometryCache.has(key) && !state.pending.has(key),
    );
    for (let offset = 0; offset < missing.length; offset += 200) {
      const group = missing.slice(offset, offset + 200);
      const batch = api(
        "/api/geometry",
        post({ legs: group.map(([, pair]) => pair) }),
      )
        .then((response) => {
          for (const leg of response.legs)
            state.geometryCache.set(JSON.stringify([leg.from, leg.to]), leg);
        })
        .finally(() => {
          for (const [key] of group) state.pending.delete(key);
        });
      for (const [key] of group) state.pending.set(key, batch);
    }
    await Promise.all(
      [...pairs.keys()].map((key) => state.pending.get(key)).filter(Boolean),
    );
  }

  /** Render objective-aware metrics, operator details, and route stops.
   * @param {PlaybackFrame} frame - Displayed cumulative algorithm state.
   * @returns {void}
   */
  function updateInspector(frame) {
    $("inspector").hidden = false;
    const result = frame.candidate || frame.current || frame.best;
    const labels = {
      initial_route: "Choose an initial route",
      ant_start: "Start an ant route",
      pheromone_update: "Update pheromone trails",
      expand: "Expand a partial path",
      insert: "Insert a stop",
      remove: "Remove a stop",
      destroy: "Choose ALNS operators",
      shake: "Shake the neighbourhood",
      replace: "Replace a stop",
      relocate: "Move a stop",
      initial_solution: "Initial solution",
      search_start: "Start search",
      search_end: "Search finished",
      result: "Final route",
      stopped: "Search stopped",
      error: "Search failed",
    };
    $("action-title").textContent =
      frame.event === "iteration"
        ? state.params.algorithm === "aco"
          ? "Complete an ant route"
          : frame.accepted
            ? "Accept candidate"
            : "Keep current route"
        : labels[frame.action || frame.event] || "Explore route";
    const shortest = state.params.algorithm === "dijkstra";
    $("iteration-label").textContent = shortest
      ? `Expanded ${frame.expanded} labels`
      : frame.iteration
        ? `Iteration ${frame.iteration} of ${state.params.iterations}`
        : "Initial construction";
    const detail = [];
    if (frame.destroy_operator)
      detail.push(
        `${frame.destroy_operator} removal, ${frame.repair_operator} repair`,
      );
    if (frame.neighbourhood)
      detail.push(
        `${frame.event === "iteration" ? "Next neighbourhood" : "Neighbourhood"} ${frame.neighbourhood}`,
      );
    if (frame.temperature !== undefined)
      detail.push(`Temperature ${frame.temperature.toFixed(2)}`);
    if (frame.affected.length)
      detail.push(`Affected: ${frame.affected.join(", ")}`);
    if (frame.queued !== undefined)
      detail.push(`${frame.queued} labels waiting`);
    if (frame.pheromone_edges !== undefined)
      detail.push(
        `Evaporate pheromone, then reinforce the best route (${frame.pheromone_edges} tracked edges)`,
      );
    $("action-detail").textContent =
      frame.message ||
      detail.join(". ") ||
      (frame.event === "result"
        ? shortest
          ? "First feasible destination path found by the time-dependent label search."
          : "Best feasible route found in this search."
        : "Selected operator action; individual trial evaluations are omitted.");
    $("decision").hidden = frame.accepted === undefined && !frame.improved_best;
    $("decision").textContent = frame.improved_best
      ? "New best"
      : frame.accepted
        ? "Accepted"
        : "Rejected";
    $("decision").className =
      "decision" + (frame.accepted === false ? " rejected" : "");
    $("route-score").textContent = result ? result.score.toFixed(2) : "—";
    $("route-time").textContent = result
      ? `${result.duration_min.toFixed(1)} min`
      : "—";
    $("best-metric-label").textContent = shortest
      ? "Destination time"
      : "Best score";
    $("best-score").textContent = frame.best
      ? shortest
        ? `${frame.best.duration_min.toFixed(1)} min`
        : frame.best.score.toFixed(2)
      : "—";
    $("stop-metric-label").textContent = shortest
      ? "Path points"
      : "Selected stops";
    $("stop-count").textContent = result
      ? shortest
        ? result.route.length
        : Math.max(0, result.route.length - 2)
      : "—";
    const complete = result?.route.at(-1) === state.params.end;
    $("route-summary").textContent =
      `${shortest && !complete ? "Partial path" : "Route stops"} (${result?.route.length || 0}${complete ? ", including endpoints" : ""})`;
    $("route-stops").replaceChildren();
    for (const id of result?.route || []) {
      const li = document.createElement("li"),
        button = document.createElement("button"),
        point = state.byId.get(id);
      button.textContent = `${id} ${point?.name || ""}`;
      button.onclick = () => {
        if (map && point) {
          map.flyTo({
            center: [point.lon, point.lat],
            zoom: 15,
            essential: false,
          });
          showPoint(point, [point.lon, point.lat]);
        }
      };
      li.append(button);
      $("route-stops").append(li);
    }
    drawProgressChart(frame, shortest);
  }

  /** Plot score improvement or Dijkstra's elapsed expansion times.
   * @param {PlaybackFrame} frame - Currently displayed algorithm state.
   * @param {boolean} shortest - Whether the selected solver minimises time.
   * @returns {void}
   */
  function drawProgressChart(frame, shortest) {
    const chart = state.frames
      .slice(0, state.index + 1)
      .filter((f) =>
        shortest
          ? f.action === "expand" && f.candidate
          : (["initial_solution", "iteration"].includes(f.event) ||
              f.action === "pheromone_update") &&
            f.best,
      )
      .map((f) =>
        shortest
          ? { x: f.expanded, y: f.candidate.duration_min }
          : { x: f.iteration, y: f.best.score },
      );
    const extent = shortest ? frame.expanded : state.params.iterations;
    $("chart-title").textContent = shortest
      ? "Expanded path time (min)"
      : "Best score by iteration";
    $("score-chart").setAttribute(
      "aria-label",
      shortest
        ? "Expanded path travel time by label count"
        : "Best route score over iterations",
    );
    $("chart-end").textContent = extent;
    if (chart.length) {
      const low = Math.min(...chart.map((f) => f.y));
      const high = Math.max(low + 1, ...chart.map((f) => f.y));
      const points = chart.map((f) => [
        4 + (252 * f.x) / Math.max(1, extent),
        70 - (62 * (f.y - low)) / (high - low),
      ]);
      $("chart-line").setAttribute(
        "points",
        points.map((p) => p.join(",")).join(" "),
      );
      $("chart-point").setAttribute("cx", points.at(-1)[0]);
      $("chart-point").setAttribute("cy", points.at(-1)[1]);
      $("chart-score").textContent = chart.at(-1).y.toFixed(2);
    } else {
      $("chart-line").setAttribute("points", "");
      $("chart-score").textContent = "—";
    }
  }

  /** Resolve road geometry and render only if this request is still current.
   * @param {number} index - Requested zero-based recorded frame index.
   * @returns {Promise<void>}
   */
  async function renderFrame(index) {
    if (index < 0 || index >= state.frames.length) return;
    const token = ++state.renderToken;
    const generation = state.generation;
    const frame = state.frames[index];
    state.busy = true;
    updateControls();
    try {
      const pairs = legPairs(frame);
      await ensureGeometry(pairs);
      if (token !== state.renderToken || generation !== state.generation)
        return;
      const missing = new Set();
      if (state.mapReady) {
        for (const name of ["best", "current", "candidate"]) {
          const route = frame[name]?.route || [];
          const signature = JSON.stringify(route);
          const features = [];
          for (let i = 1; i < route.length; i++) {
            const leg = state.geometryCache.get(
              JSON.stringify([route[i - 1], route[i]]),
            );
            if (leg?.geometry)
              features.push({
                type: "Feature",
                geometry: leg.geometry,
                properties: {},
              });
            else missing.add(`${route[i - 1]} → ${route[i]}`);
          }
          if (state.renderedRoutes[name] !== signature) {
            map
              .getSource(name)
              .setData({ type: "FeatureCollection", features });
            state.renderedRoutes[name] = signature;
          }
        }
        updateMarkers(frame.affected);
      }
      state.index = index;
      updateInspector(frame);
      $("geometry-warning").hidden = !missing.size;
      $("geometry-warning").textContent =
        `Missing road connections: ${[...missing].slice(0, 4).join(", ")}${missing.size > 4 ? "…" : ""}. No substitute lines are drawn.`;
    } catch (error) {
      if (token !== state.renderToken) return;
      if (error.status === 503)
        $("playback-state").textContent = "Loading geometry";
      else {
        state.playing = false;
        formError(
          `Route geometry could not load: ${error.message}. Resume playback to retry.`,
        );
      }
    } finally {
      if (token === state.renderToken) {
        state.busy = false;
        updateControls();
      }
    }
  }

  $("play").onclick = () => {
    if (state.index >= state.frames.length - 1 && !state.active) {
      state.index = -1;
      state.playing = true;
    } else state.playing = !state.playing;
    state.lastAdvance = 0;
    updateControls();
  };
  $("step").onclick = () => {
    state.playing = false;
    renderFrame(state.index + 1);
    updateControls();
  };
  $("timeline").oninput = (event) => {
    state.playing = false;
    renderFrame(Number(event.target.value));
    updateControls();
  };
  /** Advance playback independently from solver computation and map panning.
   * @param {number} timestamp - requestAnimationFrame time in milliseconds.
   * @returns {void}
   */
  function animate(timestamp) {
    if (
      state.playing &&
      state.graphReady &&
      !state.busy &&
      state.index < state.frames.length - 1 &&
      timestamp - state.lastAdvance >= 1000 / Number($("speed").value)
    ) {
      state.lastAdvance = timestamp;
      renderFrame(state.index + 1);
    }
    requestAnimationFrame(animate);
  }
  window.addEventListener("pagehide", () => {
    if (state.active && state.runId)
      navigator.sendBeacon(`/api/runs/${state.runId}/stop`);
  });

  /** Load metadata, set valid defaults, and begin background UI polling.
   * @returns {Promise<void>}
   */
  async function initialize() {
    initializeMap();
    if (window.matchMedia("(max-width:700px)").matches) collapse(true);
    try {
      /** @type {BootstrapPayload} */
      const data = await api("/api/bootstrap");
      state.points = data.points;
      state.byId = new Map(data.points.map((p) => [p.id, p]));
      state.ring = data.ring;
      for (const field of ["start", "end"]) {
        const options = data.points.map((p) => {
          const option = document.createElement("option");
          option.value = p.id;
          option.label = `${p.id} — ${p.name}`;
          return option;
        });
        $(field + "-options").replaceChildren(...options);
      }
      for (const [key, value] of Object.entries(data.defaults))
        $(key).value = value;
      algorithmInputs();
      $("dataset-name").textContent =
        `${data.points.length} locations · ${data.matrix}`;
      state.ready = true;
      endpointNames();
      updateControls();
      if (state.mapReady) map.getSource("ring").setData(state.ring);
      if (map) {
        if (state.mapReady) fitRoute();
        else map.once("load", fitRoute);
      }
      refreshStatus();
    } catch (error) {
      formError(
        `Locations could not load: ${error.message}. Restart the local server and reload.`,
      );
    }
    requestAnimationFrame(animate);
  }
  initialize();
})();
