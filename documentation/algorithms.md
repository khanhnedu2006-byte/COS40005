# Route-search algorithms

## Common problem and evaluation

The application searches an origin/destination matrix, not individual OSM road
nodes. A request supplies distinct known endpoints, a local departure minute,
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

The browser displays the selected operators, actual removals/insertions,
acceptance or rejection, temperature, and incumbent improvements. It does not
emit every trial insertion considered inside an operator.

## Variable neighbourhood search: VNS

VNS starts from a repaired feasible route. It shakes the route by removing a
number of stops, applies randomized repair, then tries single-stop replacement
and relocation in one local-search pass. A better candidate becomes the new
incumbent and resets the neighbourhood size to one. Otherwise the size cycles
through one to four.

Playback distinguishes the neighbourhood used to shake a route from the next
neighbourhood selected after an iteration. Replacement and relocation events
show successful local improvements, rather than every rejected local trial.

## Ant Colony optimization: ACO

`AntColony` is a compact pheromone-guided insertion search. This implementation
constructs one ant route per iteration, beginning with the same feasible initial
route. It inserts feasible improving POIs until no improving insertion remains.

Insertion weights combine pheromone on the two newly introduced directed
point-pair edges with score gain relative to extra travel time. These are OD
edges between stops, not individual OSM road segments. After construction,
pheromone values are multiplied by 0.9 with a minimum of 0.1; edges in the
best-ever route receive an additional 1.0.

Playback shows each ant's start, selected insertions, the pheromone update,
the completed ant route, and any new best route. The map illustrates the
reinforced route; it is not a heatmap of every pheromone value.

## Dijkstra-style time-dependent baseline

`shortest_route` prioritizes pending paths by elapsed travel time. Its labels
distinguish the current point and arrival minute; expansions use the departure
slot of the next leg. Paths that revisit a point or exceed the budget are
excluded. The first feasible destination path found is returned.

This baseline seeks a short travel-time route rather than maximizing POI score.
It is deterministic and does not use iterations or a random seed. Because
labels group arrivals by minute and waiting is absent, it is an approximation
for time-dependent travel times, not a general exact-optimality guarantee.

The browser shows each popped label's partial path, elapsed time and queue size.
The initial start-only prefix has zero travel time. Its chart plots expanded
path time against label count; the other algorithms plot best score by iteration.
Default CLI progress remains lightweight unless detailed tracing is requested.

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
Record completion, score, duration, computation time and seed. Playback speed
controls presentation and must not be treated as solver runtime.
