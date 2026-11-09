/**
 * Browser-side documentation of the JSON contracts in Web/types.py.
 *
 * This module is imported by JSDoc only. It adds no runtime request or build
 * dependency; Python validates inputs and supplies these payloads over HTTP/SSE.
 *
 * @typedef {"alns" | "vns" | "aco" | "dijkstra"} AlgorithmName
 *
 * @typedef {Object} SearchParameters
 * @property {AlgorithmName} algorithm - Selected solver.
 * @property {string} start - Matrix start ID.
 * @property {string} end - Distinct destination ID.
 * @property {string} departure - Local HH:MM departure time.
 * @property {number} budget_min - Positive finite travel-time budget.
 * @property {number} iterations - Non-negative count; unused by Dijkstra.
 * @property {number} seed - Random seed; unused by Dijkstra.
 *
 * @typedef {Object} RouteSnapshot
 * @property {string[]} route - Ordered IDs, possibly a partial expanded path.
 * @property {number} score - Unique selected/passed POI score.
 * @property {number} duration_min - Elapsed travel time in minutes.
 * @property {string[]} collected - Unique credited POI IDs.
 *
 * @typedef {Object} PointMetadata
 * @property {string} id - Stable matrix ID.
 * @property {string} name - Configured display name, falling back to the ID.
 * @property {number} lat - Latitude in degrees.
 * @property {number} lon - Longitude in degrees.
 * @property {string} kind - Hub, POI, or custom point type.
 * @property {number} score - Matrix score, zero for hubs.
 *
 * @typedef {Object} SearchEvent
 * @property {string} event - Lifecycle or operator-event discriminator.
 * @property {number} sequence - One-based run-local SSE cursor.
 * @property {number} version - Event protocol version.
 * @property {AlgorithmName} algorithm - Solver producing the event.
 * @property {number} [iteration] - Current heuristic iteration.
 * @property {number} [iterations] - Configured heuristic iteration count.
 * @property {RouteSnapshot} [best] - Best complete route found so far.
 * @property {number} [best_score] - Scalar incumbent score for CLI consumers.
 * @property {number} [best_duration_min] - Scalar incumbent travel time.
 * @property {RouteSnapshot} [current] - Current accepted/constructed route.
 * @property {RouteSnapshot} [candidate] - Candidate or expanded partial path.
 * @property {string} [action] - Selected operator action.
 * @property {boolean} [accepted] - ALNS/VNS acceptance decision.
 * @property {boolean} [improved_best] - Whether a new incumbent was found.
 * @property {number} [expanded] - Dijkstra labels expanded so far.
 * @property {number} [queued] - Remaining priority-queue labels.
 * @property {boolean} [found] - Whether a Dijkstra destination was reached.
 * @property {number | null} [duration_min] - Destination duration at search end.
 * @property {number} [neighbourhood] - VNS neighbourhood size.
 * @property {number} [temperature] - ALNS acceptance temperature.
 * @property {string} [destroy_operator] - Selected ALNS removal method.
 * @property {string} [repair_operator] - Selected ALNS insertion method.
 * @property {number} [pheromone_edges] - Number of ACO tracked edges.
 * @property {number} [evaporation] - Multiplicative pheromone retention factor.
 * @property {string} [message] - Terminal error explanation.
 *
 * @typedef {Omit<SearchEvent, "candidate"> & {iteration: number, expanded: number,
 *   affected: string[], candidate?: RouteSnapshot | null}} PlaybackFrame
 *
 * @typedef {Object} GeometryLeg
 * @property {string} from - Origin ID.
 * @property {string} to - Destination ID.
 * @property {{type: "MultiLineString", coordinates: number[][][]} | null} geometry - Road shape, or an explicit gap.
 * @property {string | null} message - Missing-path explanation.
 *
 * @typedef {Object} GraphStatusPayload
 * @property {"loading" | "ready" | "error"} graph - Background graph state.
 * @property {string | null} message - Initialization failure explanation.
 * @property {number} graph_loads - Graph initialization count for this session.
 * @property {number | null} graph_seconds - Last graph-load duration.
 * @property {number} cached_legs - Number of cached directed pairs.
 *
 * @typedef {Object} BootstrapPayload
 * @property {PointMetadata[]} points - Available matrix locations.
 * @property {string} matrix - Selected CSV filename.
 * @property {Object} ring - Central-ring GeoJSON FeatureCollection.
 * @property {SearchParameters} defaults - Valid initial form values.
 * @property {GraphStatusPayload} status - Current background-loader state.
 *
 * @typedef {Object} BrowserState
 * @property {PointMetadata[]} points - Loaded locations.
 * @property {Map<string, PointMetadata>} byId - Marker lookup.
 * @property {PlaybackFrame[]} frames - Recorded cumulative snapshots.
 * @property {number} index - Displayed frame index, or -1 before playback.
 * @property {boolean} playing - Whether playback is advancing.
 * @property {string | null} runId - Current run ID.
 * @property {boolean} active - Whether computation is still running.
 * @property {EventSource | null} stream - Current reconnectable SSE stream.
 * @property {boolean} ready - Whether location metadata loaded.
 * @property {boolean} graphReady - Whether geometry can be requested.
 * @property {Map<string, GeometryLeg>} geometryCache - Browser leg cache.
 * @property {Map<string, Promise<void>>} pending - In-flight geometry batches.
 * @property {number} renderToken - Prevents stale frame updates.
 * @property {boolean} busy - Whether a frame awaits road geometry.
 * @property {number} generation - Prevents updates from a replaced run.
 * @property {SearchParameters | null} params - Inputs for the displayed run.
 * @property {number} lastAdvance - Last animation timestamp in milliseconds.
 * @property {boolean} mapReady - Whether custom map layers exist.
 * @property {RouteSnapshot | null} result - Downloadable final result.
 * @property {Record<string, string>} renderedRoutes - Last rendered ID sequences.
 * @property {string | null} markerSelection - Current endpoint signature.
 * @property {string | null} lastAffected - Last highlighted-stop signature.
 * @property {Object} [ring] - Central-ring GeoJSON.
 */
export {};
