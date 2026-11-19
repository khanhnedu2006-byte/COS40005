# Route-search algorithms

## Time-budgeted road solvers in the web application

Dijkstra, ALNS, VNS, ACO and both PPO solvers use the full directed base graph and configured POIs/hubs.
Travel time follows commits `76f750b` (AB estimator) and `39bb0e5` (road-edge
builder): edge seconds = length in metres × calibrated seconds/metre for its
road-class group and time slot, using `configs/estimator_params.json`. Class
mapping is shared with `ABEstimator.GROUP_OF`. Classes outside that mapping use
local-road coefficients to retain the full graph. No fixed per-connection fee is
added, matching the road-edge builder's formula.

Cached Dijkstra trees supply the fastest connection under the rates for a
single slot. If a connection may cross a slot boundary, the solver also
considers paths from the subsequent slots it could reach within the budget.
It replays each candidate using each edge's actual departure slot and keeps
the earliest arrival. For tours with waypoints it also compares the route
using the original departure-slot paths because a different waypoint arrival
can change the traffic on later legs. The destination is a sink, so a connection to a POI
cannot drive through it. PPO uses the same edge clock, including midnight
wraparound. The search compares a finite set of slot-specific candidates, so
its result is an estimate of the fastest time-dependent connection.

ALNS, VNS and ACO maximize unique POI score within a time budget, breaking ties by
narrow-road-weighted distance, then estimated time and physical distance. The
existing CSV name matcher applies 2× distance cost for unambiguous widths below
5.5 m; unknown widths remain unpenalized. These penalties do not inflate reported
travel time. Dijkstra supplies the fastest-estimated-time baseline.

All three heuristics start from the direct connection and greedily insert useful
waypoints. ALNS uses random/worst removal, greedy/randomized repair, adaptive
operator weights and annealing. VNS cycles removal sizes one through four,
repairs randomly, and tests waypoint replacement and relocation. ACO builds
one route per iteration from the direct connection, sampling feasible insertions
by score gain per second and pheromone on adjacent waypoint connections.
Pheromone evaporates each iteration and the best route reinforces its
connections. Trees and connections are built lazily and reused across trials.

Connection POI masks are cached. Scores count each POI once within 100 m of the
approximate road geometry; selected waypoints receive no automatic credit.
Distinct IDs may share a snapped node and zero-length connections are supported.
Endpoint/waypoint snapping remains limited to 300 m.

Web defaults: 240 minutes, departure 08:00, 100 iterations, seed 0. Responses
include `duration_min`, physical `distance_km`, waypoints, geometry, collected
POIs, width coverage and estimated directions. Road curves and bus restrictions
are absent from the base tables. The geometry and timing are estimates.

## Waypoint PPO

Waypoint PPO chooses an uncollected POI or finishes at the requested destination.
It shares Dijkstra connections, the traffic-slot clock, and the route renderer
with ALNS and VNS. A POI action is available when the road planner verifies
both the drive to the POI and a continuation that fits the remaining budget.
The policy earns the scores of newly passed POIs on each leg, including the
final leg. Training samples feasible hub pairs across the configured departure
slots and budgets and holds out the benchmark's ordered endpoint pairs.

```bash
python scripts/ppo_waypoint.py train --steps 100000 --seed 0
python scripts/ppo_waypoint.py infer --start H01 --end H02 --departure 08:00 --budget-min 240
python scripts/ppo_waypoint.py benchmark --report results/ppo_waypoint_benchmark.json
```

The browser and CLI use `checkpoints/ppo_waypoint.zip` by default. The
checkpoint binds the policy to the graph and ordered waypoint actions.

## Edge PPO on the full road graph

`ppo_road.py` delegates rollout collection, GAE, clipping, advantage
normalization and optimization to **stable_baselines3.PPO**. `road_ppo_env.py` defines the
Gymnasium interface, road clock, unique POI rewards, narrow-road penalties,
curriculum and compact observations. `road_ppo_policy.py` defines a masked
Torch actor and critic sharing a trainable two-layer directed GraphSAGE encoder
that processes every road node with sparse edge-list aggregation.

Each action traverses one outgoing edge, including parallel edges. Padded
slots and moves whose duration plus an optimistic return-time bound exceed
the remaining budget are masked inside the actor, during both sampling and
updates. The bound minimizes each edge's time over all slots; actual slot
changes can still cause failure. Step caps, dead ends and arrival are terminal
states, while ordinary rollout cuts bootstrap SB3's critic.

Checkpoints use SB3 ZIP serialization, preserving policy/optimizer state,
graph fingerprints, RNG state, counters and fixed held-out cases. Continuation
starts a new trip. The full-graph embeddings are cached only while collecting rollouts or
predicting without gradients. Training minibatches recompute them. Old SBX
ZIPs and `.pt` weights require retraining. See the
[pipeline walkthrough](ppo-road.md) for annotated data flow and commands.

Road inference never substitutes a Dijkstra path if the policy fails.
Successful policy edges share replay, score, geometry and instructions with
the other road solvers. Compare completion, POI score, travel time and
narrow-road exposure on identical trips; PPO's shaped reward and the
heuristics' lexicographic objectives differ.

## Legacy OD-matrix problem and evaluation


The existing CLI and native viewer search an origin/destination matrix. A request supplies distinct known endpoints, a local departure minute,
and a travel-time budget. Each directed leg uses the matrix entry for its
departure time slot. A missing or infeasible entry cannot be traversed.

`ODMatrix.evaluate` is the common source of route scores and durations. Scores
credit each selected or passed POI once; a POI used as the starting point also
counts. Waiting and dwell time are not modeled. Departure slots advance with
elapsed travel time, including wraparound at midnight.

ALNS, VNS and ACO compare `(score, -duration)`: higher score wins, with shorter
time breaking ties. They preserve a best feasible solution while exploring.
Their initial construction tries a direct route, then one-stop alternatives if
necessary. This construction can fail even when a longer feasible route exists.

## Adaptive large neighbourhood search: ALNS

ALNS constructs an initial feasible route and greedily inserts useful POIs.
Each iteration selects a removal method and a repair method, removes one to
four intermediate stops where feasible, then repairs the partial route.

- Random removal chooses among feasible stop deletions.
- Worst-stop removal keeps the highest-quality remaining route.
- Greedy repair ranks improving insertions by score gain per added time.
- Randomized repair chooses among the three leading insertion options.

An improved candidate is accepted. A lower-score candidate can also be accepted
with probability `exp(score_delta / temperature)`; the temperature decreases
toward 0.01. The best-ever route is retained separately. Operator weights use
an exponential update: 80% previous weight plus 20% outcome reward.

The browser displays only the best complete route returned after all iterations.
Operator tracing remains optional in the core solver for other consumers.

## Variable neighbourhood search: VNS

VNS starts from a repaired feasible route. It shakes the route by removing a
number of stops, applies randomized repair, then tries single-stop replacement
and relocation in one local-search pass. A better candidate becomes the new
incumbent and resets the neighbourhood size to one. Otherwise the size cycles
through one to four.

Only the completed best route is sent to the browser; neighbourhood changes
and local-search trials are not visualized.

## Ant Colony optimization: ACO

`AntColony` is a compact pheromone-guided insertion search. This implementation
constructs one ant route per iteration, beginning with the same feasible initial
route. It inserts feasible improving POIs until no improving insertion remains.

Insertion weights combine pheromone on the two newly introduced directed
point-pair edges with score gain relative to extra travel time. These are OD
edges between stops, not individual OSM road segments. After construction,
pheromone values are multiplied by 0.9 with a minimum of 0.1; edges in the
best-ever route receive an additional 1.0.

The browser displays the best completed route after computation. Ant
construction and pheromone updates remain internal to the solver.

## Dijkstra-style time-dependent baseline

`shortest_route` prioritizes pending paths by elapsed travel time. Its labels
distinguish the current point and arrival minute; expansions use the departure
slot of the next leg. Paths that revisit a point or exceed the budget are
excluded. The first feasible destination path found is returned.

This baseline seeks a short travel-time route rather than maximizing POI score.
It is deterministic and does not use iterations or a random seed. Because
labels group arrivals by minute and waiting is absent, it is an approximation
for time-dependent travel times, not a general exact-optimality guarantee.

The browser displays only the returned complete path and travel time. Expanded
partial paths are not sent to the web app. Default CLI progress remains
lightweight unless detailed tracing is requested.

There is a second use of Dijkstra in `Web/road_paths.py`: NetworkX finds a
road-length-weighted path on OSM to draw a leg. That reconstruction does not
replace this matrix-based travel-time baseline or change its reported duration.

## Other existing approaches

The greedy OD baseline selects feasible next POIs by newly collected score per
travel time, using the environment's return-time estimate. It provides a simple
comparison for learned policies and remains outside the browser selector.

The existing PPO implementation uses SBX/JAX with a masked GraphSAGE policy.
Its environment tracks the current point, remaining time, departure slot and
collected POIs. Action masks exclude currently unavailable legs and estimate
whether the destination can still be reached. That return-time estimate freezes
a slot, so later slot changes can still cause a dead end. Policies can be trained
or loaded from matrix-bound checkpoints through the CLI. See the existing
[PPO instructions](../README.md#ppo-routing-on-the-od-matrix) for training and
evaluation commands.

For fair comparisons, use identical endpoints, departure, budget and matrix.
Record completion, score, duration, computation time and seed. Road geometry
initialization and rendering time are separate from solver runtime.
