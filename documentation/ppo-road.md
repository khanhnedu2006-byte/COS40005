# Full-road-graph PPO with SB3 on CPU

This is the training pipeline behind the browser's **PPO** option. It uses
`stable_baselines3.PPO` from **Stable-Baselines3 2.9.2**, with Torch and
`device="cpu"`. The road CLI does not import JAX/SBX or compile XLA kernels.
Older SBX or `.pt` checkpoints are incompatible and rejected before
deserialization.

## Start here

From an editable checkout with dependencies installed:

```bash
python scripts/ppo_road.py train --steps 200000 --threads 4 --checkpoint checkpoints/ppo_road.zip --report results/ppo_road.json
python scripts/ppo_road.py evaluate --checkpoint checkpoints/ppo_road.zip --report results/ppo_road_eval.json
python scripts/ppo_road.py infer --checkpoint checkpoints/ppo_road.zip --start H01 --end H02 --departure 08:00 --budget-min 240
```

For a correctness smoke test rather than a useful trained model:

```bash
python scripts/ppo_road.py train --steps 8 --rollout-steps 8 --batch-size 4 --epochs 1 --hidden-dim 8 --max-episode-steps 8 --eval-count 2 --checkpoint /tmp/ppo_road_smoke.zip --report /tmp/ppo_road_smoke.json
```

There is no JAX compilation warmup. A tiny run tests the plumbing,
not route quality. JSON progress goes to stderr; stdout and `--report` contain
the final report. `--eval-count 0` skips evaluation for smoke tests.

## Which file does what?

Read the files in this order:

1. [`road_ppo_env.py`](../src/busroute/routing/road_ppo_env.py): configuration,
   graph features, road-trip rules, sampling and the Gymnasium adapter.
2. [`road_ppo_policy.py`](../src/busroute/routing/road_ppo_policy.py): two-layer
   directed GraphSAGE and a masked SB3 ActorCriticPolicy, written in Torch.
3. [`ppo_road.py`](../src/busroute/routing/ppo_road.py): construct `stable_baselines3.PPO`, call
   `learn`, save completed updates, validate checkpoints, infer and evaluate.
4. [`scripts/ppo_road.py`](../scripts/ppo_road.py): command-line arguments,
   data loading, fixed evaluation cases and JSON reports.

The library handles rollout storage, GAE, clipping, entropy regularization,
advantage normalization, gradient clipping and Adam updates. We retain the
routing environment and network definitions because ordinary PPO does not know
what a valid road action, a POI reward or a time budget means. There is no
second PPO implementation inside these modules.

## What one decision looks like

The inputs are all 93,751 nodes and 230,749 directed edges from
`data/stage1/nodes_base.parquet` and `edges_base.parquet`, plus configured hubs,
POIs and `data/final/duong_pho_clean.csv` widths. The curriculum changes trips,
never the graph size.

Suppose the current intersection has three outgoing edges. An action is slot
0, 1 or 2 in that node's edge list. It traverses one actual edge, including its
parallel-edge identity. The environment then:

1. Charges `length_m * seconds_per_metre` using that edge's departure slot.
   The next edge uses the updated clock, including midnight wraparound.
2. Credits newly passed POIs, within 100 m of the graph's approximate geometry.
   Bit masks ensure each POI earns reward once per trip.
3. Deducts travel and known narrow-road penalties.
4. Checks arrival, the budget, step cap and whether any next action is valid.

The shaped reward is:

```text
new POI score
- 0.01 * travel minutes
- 0.05 * minutes on known roads narrower than 5.5 m
+ 10 if the destination is reached
- 20 if the trip fails
```

These coefficients are configurable initial experiment settings. Unknown road
widths receive no narrow-road penalty; the report tracks unknown-width distance.
They do not change the reported physical travel time or distance.

An episode ends at the destination, or as a failure when it cannot proceed,
exceeds its budget or reaches its edge limit. These are true Gymnasium terminal
states, including the deliberate failure at the step cap. They must not be
bootstrapped as if more reward could follow. A PPO rollout ending mid-trip is
different: SB3 bootstraps the critic automatically and continues the trip.

## Why the observations remain small

K is the graph's actual maximum outgoing degree. All outgoing edges are kept;
padding fills unused slots. The flat float32 observation contains:

- Current node and destination indices: 2 values.
- Candidate destination node indices: K values.
- Edge features: K times 6 values (travel fraction, distance, known-narrow flag,
  unknown-width flag, return-time bound and uncollected POI reward).
- Trip state: 5 values plus one visited bit per POI (remaining budget fraction,
  clock sine/cosine, budget scale, and current return-time bound).
- Action mask: K values.

Total: `2 + 8*K + 5 + number_of_POIs`. Node indices are exactly representable
in float32 for this dataset. `RoadFeatures.pack` and `decode_observations`
define this explicit, compact Box layout.

The static graph is stored separately in `StaticRoadGraph`, outside the
rollout buffer. A graph encoder aggregates outgoing and incoming neighbors
using sparse normalized matrix multiplication, with O(N+E) storage rather
than a dense N*N adjacency matrix. The first layer's raw neighborhood features are immutable and
are computed once with NumPy; the learned layers still process every node and
receive gradients. Static buffers are reconstructed from policy kwargs rather
than duplicated in every rollout entry.

For each action, the actor combines current, goal and candidate embeddings
with edge and trip features. The critic uses current and goal embeddings plus
trip state. Both heads share one trainable encoder and SB3's standard optimizer.

During rollouts and inference, weights are fixed and Torch gradients are
disabled, so the policy caches the full-graph node embeddings once and reuses
them across decisions and critic bootstrapping. Clock, remaining budget,
visited POIs and masks are recomputed for each decision. SB3's training-mode
transition invalidates the cache, and every gradient-enabled minibatch performs
a fresh full-graph pass. After externally modifying weights, call
`policy.clear_embedding_cache()` before further inference.

## Masks and Dijkstra's limited role

Padding is always invalid. A real edge is masked when its immediate travel
time plus an optimistic return-time lower bound exceeds the remaining budget.
The bound uses reverse Dijkstra with each edge's minimum travel time over all
slots. For parallel edges, the minimum is used rather than their sum.

A bound can be optimistic: changing traffic slots may still cause failure.
Dijkstra never chooses the action or replaces the policy's route. Inference
reports failure if PPO does not arrive. It does not quietly substitute a
shortest path.

An all-masked synthetic/terminal sample gets a finite sentinel distribution to
avoid NaNs. The environment and inference loop reject all-masked *routing*
states before requesting an action. Masking is applied inside the actor, so it
also applies during PPO updates and `model.predict`, not just during sampling.

## Training, evaluation and continuation

Training starts with short reachable trips. Over `curriculum_steps`, sampling
increasingly mixes longer random walks, arbitrary nodes and configured
endpoints, with random departure times and time budgets. By default twenty
fixed evaluation cases are saved. Their ordered endpoint pairs are excluded
from training, regardless of departure or budget.

Reports give completion counts, successful-trip POI score, duration, physical
distance and known-narrow/unknown-width distances. A failed trip has a null
score. Short-prefix distance/time metrics for failures are diagnostic only.
Use identical trips when comparing with Dijkstra, ALNS and VNS, and report
completion before scores. Their objectives differ from PPO's shaped reward.

SB3's `learn(total_timesteps=...)` completes full rollouts, so `--steps` is a
lower bound: actual interactions round up to a multiple of `--rollout-steps`.
The minibatch size must divide rollout size. Progress records distinguish
completed rollout updates from SB3's internal epoch counter.

The callback saves atomically after each completed optimizer update. It cannot
save at `on_rollout_end`, because SB3 calls that hook *before* optimization; it
saves at the next rollout start or training end instead. The standard SB3 ZIP
contains shared policy weights, its optimizer state and static graph inputs,
plus our graph fingerprint, configuration, sampler and global Torch/NumPy/Python
RNG states, counters and held-out cases.

Continue with:

```bash
python scripts/ppo_road.py train --resume checkpoints/ppo_road.zip --steps 200000 --checkpoint checkpoints/ppo_road.zip --report results/ppo_road_resumed.json
```

The CLI inherits the saved settings, including non-default hidden width and
rewards. Conflicting overrides are rejected. `--steps` is additional work.
Resume starts a new trip using the saved sampler stream; the active trip and
rollout buffer are not restored, so it is not an exact uninterrupted replay.

Graph ordering, coordinates, POIs, widths or travel-time changes invalidate the
checkpoint. The checkpoint format now records backend `sb3` and version 3.
Previous SBX road ZIPs, legacy Torch `.pt` weights and OD-matrix ZIPs require
retraining; they are rejected before model deserialization. In the browser choose **PPO** and enter the road ZIP
path. The successful policy edge sequence goes through the same renderer as
other road solvers, producing road instructions and highlighted passed POIs.

## CPU performance and thread controls

The road pipeline always constructs and loads SB3 models with `device="cpu"`.
`--threads` controls Torch intra-operation threads (default 4); try 1, 2 or 4
for your CPU before starting a long run. More threads can add overhead to the
small policy heads and sparse graph operations. Full-graph gradient passes,
Python/SciPy environment steps and destination-bound searches still cost time.

A short local CPU comparison on the full 93,751-node / 230,749-edge graph used
32 interactions, two 16-step rollouts, minibatches of 8, one epoch, hidden width
32, seed 0 and no evaluation:

- Previous SBX implementation: 15.77 s end to end; 2.16 s between update records.
- New SB3 implementation with four Torch threads: 5.96 s end to end; 0.37 s
  between update records.

These are single short implementation comparisons, including startup, graph
loading, library overhead and checkpoint writes. The SB3 implementation also
shares its encoder and caches rollout embeddings, while SBX used separate
encoders. Different stochastic trajectories can change environment costs.
This is not an isolated JAX-versus-Torch benchmark, a long-run throughput
promise, or evidence that the policy learns better routes. Measure longer runs
on your target settings before extrapolating training duration.

See the official [custom-policy guide](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)
for SB3's extension points. The core PPO loop is supplied by SB3; there is no
handwritten replacement optimizer/trainer in this project.

## Types and API documentation

The environment, policy, training helpers and CLI have annotated parameters and
return values with Google-style `Args`, `Returns`, `Raises` and `Attributes`
sections where applicable. Array documentation gives both dtype and shape:
`N` is the node count, `E` the directed-edge count, `K` the maximum outgoing
degree, `P` the POI count, and `B` a neural-network batch size. Node/edge indices
are table row positions; action indices are local outgoing slots.

`RoadObservation` describes a decision before packing. `TrainingRecord`,
`ScenarioResult`, `TrainingSummary` and `EvaluationReport` describe progress and
JSON report fields. Budgets/departures in scenario inputs use minutes;
`RoadEnv.elapsed_s` and return-time bounds use seconds. Failed evaluations have
`score=None`, while their distance/time fields describe the partial attempt.
The existing web `RouteSnapshot` contract is referenced only for type checking,
so importing the CLI does not load the web application.

The native SB3/JSON checkpoint metadata and NumPy bit-generator state remain
heterogeneous dictionaries. A small typing protocol documents SB3's dynamic
`road_metadata` attribute; localized casts bridge those existing
interfaces without changing model classes or adding runtime validation.

Run static checks with the project's interpreter and installed development tools:

```bash
ty check src/busroute/routing/ppo_road.py src/busroute/routing/road_ppo_env.py src/busroute/routing/road_ppo_policy.py scripts/ppo_road.py --python .venv
ruff check --select D --config 'lint.pydocstyle.convention="google"' src/busroute/routing/ppo_road.py src/busroute/routing/road_ppo_env.py src/busroute/routing/road_ppo_policy.py scripts/ppo_road.py
```
