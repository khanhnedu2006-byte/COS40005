/* Display only the completed route; computation stays in an isolated process. */
"use strict";
/**
 * @typedef {import("./contracts.js").AlgorithmName} AlgorithmName
 * @typedef {import("./contracts.js").SearchParameters} SearchParameters
 * @typedef {import("./contracts.js").RouteSnapshot} RouteSnapshot
 * @typedef {import("./contracts.js").PointMetadata} PointMetadata
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
  /** @type {BrowserState} */
  const state = {
    points: [],
    byId: new Map(),
    runId: null,
    active: false,
    ready: false,
    graphReady: false,
    geometryCache: new Map(),
    pending: new Map(),
    renderToken: 0,
    generation: 0,
    params: null,
    mapReady: false,
    initialFit: false,
    result: null,
    guideOpen: false,
    guideStopIndex: 0,
    markerSelection: null,
  };
  let map;
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

  /** Update endpoint colors without rebuilding the map.
   * @returns {void}
   */
  function updateMarkers() {
    if (!state.mapReady) return;
    const start =
      state.guideOpen && state.result
        ? state.result.route[0]
        : $("start").value;
    const end =
      state.guideOpen && state.result
        ? state.result.route.at(-1)
        : $("end").value;
    const features = state.points.map((p) => ({
      type: "Feature",
      id: p.id,
      geometry: { type: "Point", coordinates: [p.lon, p.lat] },
      properties: {
        id: p.id,
        name: p.name,
        label: `${p.id}  ${p.name}`,
        kind: p.kind,
        endpoint: p.id === start ? "start" : p.id === end ? "end" : "none",
      },
    }));
    const selection = JSON.stringify([start, end, state.points.length]);
    if (state.markerSelection !== selection) {
      map.getSource("points").setData({ type: "FeatureCollection", features });
      state.markerSelection = selection;
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
    map.addSource("route", { type: "geojson", data: EMPTY() });
    for (const [id, color, width] of [
      ["route-outline", "#ffffff", 9],
      ["route-line", "#3d5fd9", 5],
    ]) {
      map.addLayer({
        id,
        type: "line",
        source: "route",
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": color, "line-width": width },
      });
    }
    map.addSource("points", { type: "geojson", data: EMPTY() });
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
    map.addSource("route-stops", { type: "geojson", data: EMPTY() });
    map.addLayer({
      id: "route-stops",
      type: "circle",
      source: "route-stops",
      paint: {
        "circle-radius": 10,
        "circle-color": "#3d5fd9",
        "circle-stroke-color": "white",
        "circle-stroke-width": 2,
      },
    });
    map.addLayer({
      id: "route-stop-numbers",
      type: "symbol",
      source: "route-stops",
      layout: {
        "text-field": ["get", "number"],
        "text-font": ["Noto Sans Regular"],
        "text-size": 11,
        "text-allow-overlap": true,
      },
      paint: { "text-color": "white" },
    });
    map.addSource("guide-focus", { type: "geojson", data: EMPTY() });
    map.addLayer({
      id: "guide-focus",
      type: "circle",
      source: "guide-focus",
      paint: {
        "circle-radius": 13,
        "circle-color": "#16324f",
        "circle-stroke-color": "white",
        "circle-stroke-width": 3,
      },
    });
    map.addLayer({
      id: "guide-focus-number",
      type: "symbol",
      source: "guide-focus",
      layout: {
        "text-field": ["get", "number"],
        "text-font": ["Noto Sans Regular"],
        "text-size": 12,
        "text-allow-overlap": true,
      },
      paint: { "text-color": "white" },
    });
    state.mapReady = true;
    state.markerSelection = null;
    updateMarkers();
    updateGuideFocus();
    renderFinalRoute();
    fitInitialLocations();
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
      map.on("click", "route-stops", (event) =>
        selectGuideStop(Number(event.features[0].properties.index)),
      );
      map.on("mouseenter", "route-stops", () => {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", "route-stops", () => {
        map.getCanvas().style.cursor = "";
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

  /** Reserve visible map space around the input panel and final itinerary.
   * @returns {{top: number, bottom: number, left: number, right: number}}
   */
  function routePadding() {
    if (window.matchMedia("(max-width:700px)").matches)
      return {
        top: 90,
        bottom: state.guideOpen
          ? $("route-guide").getBoundingClientRect().height + 45
          : 50,
        left: 35,
        right: 35,
      };
    return {
      top: 90,
      bottom: 60,
      left: Math.min(380, innerWidth * 0.34),
      right: Math.min(state.guideOpen ? 440 : 60, innerWidth * 0.34),
    };
  }

  /** Fit the displayed route, or all points before a run, on explicit request.
   * @returns {void}
   */
  function fitRoute() {
    if (!map || !state.ready) return;
    const route = state.result;
    const points = route
      ? route.route.map((id) => state.byId.get(id)).filter(Boolean)
      : state.points;
    if (!points.length) return;
    const bounds = new maplibregl.LngLatBounds();
    points.forEach((p) => bounds.extend([p.lon, p.lat]));
    map.fitBounds(bounds, {
      padding: routePadding(),
      maxZoom: 15,
      duration: window.matchMedia("(prefers-reduced-motion:reduce)").matches
        ? 0
        : 500,
    });
  }
  $("fit").onclick = fitRoute;

  /** Set the initial viewport once metadata and map layers are ready.
   * @returns {void}
   */
  function fitInitialLocations() {
    if (!state.ready || !state.mapReady || state.initialFit) return;
    state.initialFit = true;
    if (!state.result) fitRoute();
  }

  /** Build the completed itinerary and its matrix-derived route summary.
   * @param {RouteSnapshot} result - Successfully completed final route.
   * @returns {void}
   */
  function buildRouteGuide(result) {
    const first = state.byId.get(result.route[0]);
    const last = state.byId.get(result.route.at(-1));
    $("guide-endpoints").textContent =
      `${first?.name || result.route[0]} to ${last?.name || result.route.at(-1)}`;
    $("guide-duration").textContent = `${result.duration_min.toFixed(1)} min`;
    $("guide-count").textContent = String(result.route.length);
    $("guide-score").textContent = result.score.toFixed(2);
    $("guide-stops").replaceChildren();
    result.route.forEach((id, index) => {
      const point = state.byId.get(id);
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "guide-stop";
      button.dataset.stopIndex = String(index);
      button.dataset.pointId = id;
      const number = document.createElement("span");
      number.className = "guide-stop-number";
      number.textContent = String(index + 1);
      number.setAttribute("aria-hidden", "true");
      const copy = document.createElement("span");
      copy.className = "guide-stop-copy";
      const role = document.createElement("span");
      role.className = "guide-stop-role";
      role.textContent =
        index === 0
          ? "Start"
          : index === result.route.length - 1
            ? "Destination"
            : `Stop ${index + 1}`;
      const name = document.createElement("strong");
      name.textContent = point?.name || id;
      const detail = document.createElement("small");
      detail.textContent = `${id} · ${point?.kind === "hub" ? "Hub" : "Point of interest"}`;
      copy.append(role, name, detail);
      button.append(number, copy);
      button.onclick = () => selectGuideStop(index);
      item.append(button);
      $("guide-stops").append(item);
    });
    state.guideStopIndex = 0;
    selectGuideStop(0, false);
    const params = state.params;
    $("guide-context").textContent =
      `${params.algorithm.toUpperCase()} · Departs ${params.departure} · ${params.budget_min} min budget`;
    $("guide-collected").textContent =
      `${result.collected.length} unique POIs credited, including those passed along the way.`;
  }

  /** Update the selected itinerary marker without changing road geometry.
   * @returns {void}
   */
  function updateGuideFocus() {
    if (!state.mapReady) return;
    const point =
      state.guideOpen && state.result
        ? state.byId.get(state.result.route[state.guideStopIndex])
        : null;
    map.getSource("guide-focus").setData(
      point
        ? {
            type: "FeatureCollection",
            features: [
              {
                type: "Feature",
                geometry: {
                  type: "Point",
                  coordinates: [point.lon, point.lat],
                },
                properties: { number: String(state.guideStopIndex + 1) },
              },
            ],
          }
        : EMPTY(),
    );
  }

  /** Select an itinerary stop and optionally move the map to its location.
   * @param {number} index - Zero-based final-route stop index.
   * @param {boolean} [moveMap=true] - Whether to focus the chosen map location.
   * @returns {void}
   */
  function selectGuideStop(index, moveMap = true) {
    if (!state.result || index < 0 || index >= state.result.route.length)
      return;
    state.guideStopIndex = index;
    const buttons = $("guide-stops").querySelectorAll(".guide-stop");
    buttons.forEach((button, i) => {
      if (i === index) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });
    $("guide-previous").disabled = index === 0;
    $("guide-next").disabled = index === state.result.route.length - 1;
    $("guide-progress").textContent =
      `${index + 1} of ${state.result.route.length}`;
    updateGuideFocus();
    if (moveMap) {
      buttons[index]?.scrollIntoView({ block: "nearest" });
      const point = state.byId.get(state.result.route[index]);
      if (map && state.mapReady && point) {
        map.flyTo({
          center: [point.lon, point.lat],
          zoom: 15,
          padding: routePadding(),
          duration: window.matchMedia("(prefers-reduced-motion:reduce)").matches
            ? 0
            : 500,
        });
      }
    }
  }

  /** Reveal the completed itinerary and keep map layers tied to its inputs.
   * @param {boolean} open - Whether the final result panel is visible.
   * @returns {void}
   */
  function setGuideVisible(open) {
    state.guideOpen = open && !!state.result;
    $("route-guide").hidden = !state.guideOpen;
    updateMarkers();
    updateGuideFocus();
    if (state.guideOpen && window.matchMedia("(max-width:700px)").matches)
      collapse(true);
  }
  $("guide-edit").onclick = () => {
    collapse(false);
    $("start").focus();
  };
  $("guide-previous").onclick = () => selectGuideStop(state.guideStopIndex - 1);
  $("guide-next").onclick = () => selectGuideStop(state.guideStopIndex + 1);
  $("retry-geometry").onclick = () => renderFinalRoute();
  window.matchMedia("(max-width:700px)").addEventListener("change", (event) => {
    if (event.matches && state.guideOpen) collapse(true);
  });

  /** Poll graph readiness while keeping the map and form interactive.
   * @returns {Promise<void>}
   */
  async function refreshStatus() {
    try {
      /** @type {GraphStatusPayload} */
      const status = await api("/api/status");
      state.graphReady = status.graph === "ready";
      if (state.graphReady && state.result) renderFinalRoute();
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
      if (status.graph === "error") {
        state.graphError = status.message || "Road graph unavailable";
        if (state.result) renderFinalRoute();
      }
      if (status.graph === "loading") window.setTimeout(refreshStatus, 1500);
    } catch (error) {
      $("graph-status").textContent = "Local server disconnected";
      formError(error.message + ". Check that the local server is running.");
      window.setTimeout(refreshStatus, 3000);
    }
  }

  /** Reflect computation and final-result availability in the controls.
   * @returns {void}
   */
  function updateControls() {
    $("run").disabled = !state.ready || state.active;
    $("stop").disabled = !state.active || !state.runId;
    $("run").textContent = state.active ? "Finding route…" : "Find route";
    $("run-form").setAttribute("aria-busy", String(state.active));
  }

  /** Poll only job status; successful completion reveals exactly one route.
   * @param {string} runId - Accepted run to observe.
   * @returns {Promise<void>}
   */
  async function pollRun(runId) {
    if (state.runId !== runId || !state.active) return;
    try {
      /** @type {import("./contracts.js").RunStatusPayload} */
      const outcome = await api(`/api/runs/${runId}`);
      if (state.runId !== runId || !state.active) return;
      if (outcome.status === "running") {
        $("search-state").textContent =
          `${state.params.algorithm.toUpperCase()} is finding your route…`;
        window.setTimeout(() => pollRun(runId), 350);
        return;
      }
      state.active = false;
      if (outcome.status === "result") {
        state.result = outcome.result;
        buildRouteGuide(state.result);
        setGuideVisible(true);
        $("download").href = `/api/runs/${runId}/result`;
        $("search-state").textContent = "Route ready";
        $("guide-title").focus({ preventScroll: true });
        renderFinalRoute();
      } else {
        $("search-state").textContent =
          outcome.status === "stopped" ? "Search stopped" : "No route found";
        if (outcome.status === "error") {
          formError(outcome.message);
          collapse(false);
        }
      }
      updateControls();
    } catch (error) {
      if (state.runId !== runId || !state.active) return;
      $("search-state").textContent = "Connection interrupted — retrying…";
      window.setTimeout(() => pollRun(runId), 1500);
    }
  }

  $("run-form").onsubmit = async (event) => {
    event.preventDefault();
    if (state.active) return;
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
    state.active = true;
    state.runId = null;
    updateControls();
    $("search-state").textContent = "Starting route search…";
    try {
      const response = await api("/api/runs", post(params));
      state.generation++;
      state.renderToken++;
      state.params = params;
      state.runId = response.id;
      state.result = null;
      state.guideStopIndex = 0;
      setGuideVisible(false);
      $("guide-stops").replaceChildren();
      $("guide-map-note").hidden = true;
      $("guide-map-note").textContent = "";
      $("retry-geometry").hidden = true;
      $("download").removeAttribute("href");
      if (state.mapReady) {
        map.getSource("route").setData(EMPTY());
        map.getSource("route-stops").setData(EMPTY());
      }
      updateControls();
      pollRun(state.runId);
    } catch (error) {
      state.active = false;
      $("search-state").textContent = "Search could not start";
      formError(error.message);
      updateControls();
    }
  };
  $("stop").onclick = async () => {
    if (!state.active || !state.runId) return;
    $("stop").disabled = true;
    try {
      await api(`/api/runs/${state.runId}/stop`, { method: "POST" });
      pollRun(state.runId);
    } catch (error) {
      formError(error.message);
      updateControls();
    }
  };

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

  /** Render only the completed route, reusing cached directed road legs.
   * @returns {Promise<void>}
   */
  async function renderFinalRoute() {
    if (!state.result) return;
    if (!state.mapReady) {
      $("guide-map-note").hidden = false;
      $("guide-map-note").textContent =
        "Waiting for the map. Your final itinerary is ready; use Retry map if a loading error appears.";
      return;
    }
    const result = state.result;
    const generation = state.generation;
    const token = ++state.renderToken;
    const pairs = new Map();
    for (let i = 1; i < result.route.length; i++) {
      const pair = [result.route[i - 1], result.route[i]];
      pairs.set(JSON.stringify(pair), pair);
    }
    map.getSource("route-stops").setData({
      type: "FeatureCollection",
      features: result.route.map((id, index) => {
        const point = state.byId.get(id);
        return {
          type: "Feature",
          geometry: { type: "Point", coordinates: [point.lon, point.lat] },
          properties: { number: String(index + 1), index },
        };
      }),
    });
    $("guide-map-note").hidden = false;
    $("retry-geometry").hidden = true;
    if (!state.graphReady) {
      $("guide-map-note").textContent = state.graphError
        ? `Road geometry unavailable: ${state.graphError}. Stop locations and route totals are still available.`
        : "Preparing road geometry… You can already browse the final stops.";
      return;
    }
    $("guide-map-note").textContent = "Loading final road path…";
    try {
      await ensureGeometry(pairs);
      if (generation !== state.generation || token !== state.renderToken)
        return;
      const features = [];
      const missing = [];
      for (const [key, pair] of pairs) {
        const leg = state.geometryCache.get(key);
        if (leg?.geometry)
          features.push({
            type: "Feature",
            geometry: leg.geometry,
            properties: {},
          });
        else missing.push(pair.join(" → "));
      }
      map.getSource("route").setData({ type: "FeatureCollection", features });
      $("guide-map-note").hidden = !missing.length;
      $("guide-map-note").textContent = missing.length
        ? `Road path unavailable for ${missing.join(", ")}. Gaps are not drawn; stop order and matrix totals remain valid.`
        : "";
    } catch (error) {
      if (generation !== state.generation || token !== state.renderToken)
        return;
      $("guide-map-note").textContent =
        `Road path could not load: ${error.message}. The final stop order remains available.`;
      $("retry-geometry").hidden = false;
    }
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
      fitInitialLocations();
      refreshStatus();
    } catch (error) {
      formError(
        `Locations could not load: ${error.message}. Restart the local server and reload.`,
      );
    }
  }
  initialize();
})();
