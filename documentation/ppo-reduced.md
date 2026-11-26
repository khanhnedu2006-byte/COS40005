# PPO on the reduced road graph

## 1. Graph reduction (`src/busroute/graph/reduce.py`, `scripts/reduce_graph.py`)
`python scripts/reduce_graph.py` reads `data/stage1/*_base.parquet` and writes `data/stage1_reduced/{nodes,edges}_reduced.parquet`.

| Step | Nodes |
|---|---|
| original | 93,751 |
| after pruning dead ends | 62,424 |
| after contracting pass-through nodes (same road class + name) | 44,606 (47.6%) |
| same, relaxed (`--relaxed`) | 43,603 (46.5%) |

Both steps are lossless for shortest paths. Kept nodes keep exactly the same pairwise travel distances (checked on 150 sources, error 0 m). The graph stays strongly connected. POI/hub snap nodes are never removed. Merged edges record the dropped nodes in `via_nodes` so geometry can be redrawn. Going lower than ~46% would require lossy steps such as dropping minor roads, so it was not done.

## 2. PPO (`ppo_env.py`, `ppo_train.py`, `scripts/ppo_train.py`)
- **Episode** = one trip **POI A → POI B** (sampled from snapped POIs with distinct road nodes, logged by POI name, with random departure time and time budget). **Step** = choose one outgoing edge (max 5).
- **Road width**: loaded from `data/final/duong_pho_clean.csv` using the same normalized road-name lookup as `RoadGraph` (`NARROW_WIDTH_M = 5.5` m for a 45-seat tour bus). Each candidate edge exposes a narrow-road flag and normalized width in the observation (`EDGE_FEATURES = 9`).
- **Reward per step**:
  $$r_t = -0.01 \cdot \Delta t_{\text{min}} + 0.02 \cdot (\text{dist\_poi}(s_t) - \text{dist\_poi}(s_{t+1}))_{\text{min}} + r_{\text{width}}(e_t) - 0.02 \cdot \mathbf{1}_{\text{revisit}} + \text{poi\_score} + 1.0 \cdot \mathbf{1}_{\text{reached B}} - 5.0 \cdot \mathbf{1}_{\text{failed}}$$
  where the road-width term $r_{\text{width}}(e_t)$ is strictly non-positive:
  - $w \ge 5.5\text{ m}$: $0$
  - $w < 5.5\text{ m}$: $-1.0 \cdot \ell_{\text{km}} \cdot \left(1 + \frac{5.5 - w}{5.5}\right)$
  - unknown width: $-0.05 \cdot \ell_{\text{km}}$
- **Action mask** (`MaskablePPO`): edges that make it impossible to still reach B in budget are hidden, so every trip ends validly.
- **Learning (episode-batch PPO)**: hierarchy **epoch > episode > step**. Each epoch (1) runs `episodes_per_epoch` complete episodes (each made of steps) with the network frozen, storing obs/action/mask/log-prob/value/reward per step; (2) computes GAE advantages per episode, walking backwards from the last step so late POI rewards credit earlier decisions; (3) merges all steps into one batch and learns from it in one go (`update_passes` of PPO-clip minibatch updates), then discards it. The loop is explicit in `ppo_train.py` (`run_episode`, `compute_gae`, `update_policy`); only the network/optimizer come from sb3-contrib's `MaskablePPO`.
- Baselines in the evaluation: `random`, `shortest` (straight to B), `greedy` (heuristic using the same observation features, including narrow-road avoidance).

## 3. CLI parameters (`scripts/ppo_train.py`)

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `100` | Number of training epochs (each collects a batch of complete episodes then updates once) |
| `--episodes-per-epoch` | `16` | Complete POI → POI episodes collected per epoch |
| `--update-passes` | `6` | Gradient passes over the collected batch in each epoch |
| `--batch-size` | `256` | Minibatch size during PPO-clip updates |
| `--seed` | `0` | RNG seed for model and scenario sampling |
| `--eval-count` | `30` | Fixed evaluation episodes per baseline/PPO (`0` skips evaluation) |
| `--checkpoint` | `checkpoints/ppo_reduced.zip` | Output `.zip` path for the trained model |
| `--report` | `results/ppo_reduced.json` | Output JSON path for evaluation metrics |
| `-v`, `--verbose` | `False` | Print every single step of every episode (from/to node or POI name, road name, width, rewards, remaining budget) |

```bash
python scripts/ppo_train.py --epochs 100                                           # train + evaluate
python scripts/ppo_train.py --epochs 3 --episodes-per-epoch 4 --eval-count 5       # quick smoke test
python scripts/ppo_train.py --epochs 1 --episodes-per-epoch 1 --eval-count 0 -v    # inspect every step
```

## 4. Visualization (`src/busroute/viz/ppo_viz.py`, `scripts/visualize_ppo.py`)

`scripts/visualize_ppo.py` renders an interactive **MapLibre GL JS** map (reusing the locally bundled `Web/assets/vendor/maplibre-gl.{css,js}` and OpenFreeMap vector tiles, with basemap switching and offline fallback) plus a static 3-panel PNG summary. It replays an episode step-by-step from `checkpoints/ppo_reduced.zip` alongside baselines (`greedy`, `shortest`), reconstructing curved road geometry from `via_nodes` and coloring each edge by road width (`green` ≥ 5.5 m, `red` < 5.5 m, `amber` unknown).

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/ppo_reduced.zip` | Trained `MaskablePPO` `.zip` checkpoint |
| `--start`, `--end` | random (`--seed`) | Optional start/end POI IDs (e.g. `P001`, `P010`) |
| `--seed` | `0` | Scenario RNG seed |
| `--departure`, `--budget-min` | random (`--seed`) | Optional departure time (`HH:MM`) and budget (minutes) |
| `--policies` | `ppo,greedy,shortest` | Comma-separated policies to replay and compare |
| `--benchmark` | `results/ppo_reduced.json` | Evaluation JSON for benchmark comparison |
| `--out` | `results/ppo_visualization.html` | Interactive HTML replay dashboard |
| `--png` | `results/ppo_visualization.png` | Static 3-panel Matplotlib summary figure |
| `--no-png` | `False` | Skip PNG output |

```bash
python scripts/visualize_ppo.py
python scripts/visualize_ppo.py --start P001 --end P010 --seed 0 --out results/ppo_p001_p010.html
```

