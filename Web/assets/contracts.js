/**
 * Browser-side documentation of the JSON contracts in Web/types.py.
 *
 * This module is imported by JSDoc only. It adds no runtime request or build
 * dependency; Python validates inputs and supplies these payloads over HTTP.
 *
 * @typedef {"dijkstra" | "alns" | "vns" | "aco" | "ppo" | "ppo_waypoint"} AlgorithmName
 *
 * @typedef {Object} SearchParameters
 * @property {AlgorithmName} algorithm - Selected solver.
 * @property {string} start - Matrix start ID.
 * @property {string} end - Distinct destination ID.
 * @property {number} budget_min - Positive estimated travel-time budget in minutes.
 * @property {string} departure - Local departure time HH:MM.
 * @property {number} iterations - Non-negative count for ALNS, VNS and ACO.
 * @property {number} seed - Random seed for ALNS, VNS and ACO.
 * @property {string | null} [checkpoint] - Checkpoint path for the chosen PPO solver.
 *
 * @typedef {Object} RouteSnapshot
 * @property {string[]} route - Ordered final-route IDs from start through destination.
 * @property {number} score - Unique selected/passed POI score.
 * @property {number} distance_km - Sum of road-edge lengths.
 * @property {number} duration_min - Estimated travel time in minutes.
 * @property {number} narrow_m - Length of confidently matched narrow roads.
 * @property {number} unknown_width_m - Length without an unambiguous width.
 * @property {{type: "LineString", coordinates: number[][]}} geometry - Approximate node-to-node geometry.
 * @property {{action: string, road: string, distance_m: number, coordinate: number[], coordinates: number[][]}[]} instructions - Estimated road changes.
 * @property {string[]} collected - Unique credited POI IDs.
 *
 * @typedef {Object} PointMetadata
 * @property {string} id - Stable matrix ID.
 * @property {string} name - Configured display name, falling back to the ID.
 * @property {number} lat - Latitude in degrees.
 * @property {number} lon - Longitude in degrees.
 * @property {string} kind - Hub, POI, or custom point type.
 * @property {number} score - Configured score, zero for hubs.
 * @property {boolean} [routing_available] - Within the road-node snapping limit.
 * @property {number} [snap_distance_m] - Distance to the nearest road node.
 *
 * @typedef {Object} RunStatusPayload
 * @property {"running" | "result" | "error" | "stopped"} status - Computation state.
 * @property {RouteSnapshot | null} result - Completed route after success.
 * @property {string | null} message - Failure explanation.
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
 * @property {PointMetadata[]} points - Loaded matrix locations.
 * @property {Map<string, PointMetadata>} byId - Marker lookup.
 * @property {string | null} runId - Current run ID.
 * @property {boolean} active - Whether computation is running.
 * @property {boolean} ready - Whether location metadata loaded.
 * @property {boolean} graphReady - Whether geometry can be requested.
 * @property {string | null} [graphError] - Graph initialization failure.
 * @property {Map<string, GeometryLeg>} geometryCache - Browser leg cache.
 * @property {Map<string, Promise<void>>} pending - In-flight geometry batches.
 * @property {number} renderToken - Prevents stale geometry updates.
 * @property {number} generation - Prevents updates from a replaced run.
 * @property {SearchParameters | null} params - Inputs for the displayed run.
 * @property {boolean} mapReady - Whether custom map layers exist.
 * @property {boolean} initialFit - Whether the initial location bounds were set.
 * @property {RouteSnapshot | null} result - The only route displayed.
 * @property {boolean} guideOpen - Whether the final itinerary is visible.
 * @property {number} guideStopIndex - Selected final-route stop, zero-based.
 * @property {number | null} roadStepIndex - Selected road instruction.
 * @property {string | null} markerSelection - Current endpoint signature.
 * @property {Object} [ring] - Central-ring GeoJSON.
 */
export {};
