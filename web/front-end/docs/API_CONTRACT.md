# API contract (front-end ⇄ back-end)

The front-end never holds the TomTom or Goong key. Keys live in the back-end `.env`.
Set `VITE_API_BASE` (e.g. `/`) to leave mock mode.

## GET /api/geocode?q=<text>
Returns `[{ "label": "...", "lat": 16.05, "lng": 108.2, "id": "H01"? , "kind": "Hub|POI|Address"? }]`
(back-end proxies Goong Autocomplete).

## POST /api/plan
Request:
```json
{ "start": {"lat":0,"lng":0,"label":"","id":"H01"}, "end": {"lat":0,"lng":0,"label":""},
  "departAt": "08:00", "budgetMin": 180 }
```
Response:
```json
{ "slot": "morning", "directMin": 33.0,
  "validation": { "issues": [ {"level":"error|warn|info","text":"..."} ] },
  "routes": [ {
    "algorithm": "ppo_gnn|greedy|shortest", "label": "AI route (PPO + GNN)",
    "feasible": true, "reason": "(if infeasible)", "totalMin": 172.4, "budgetMin": 180,
    "totalScore": 154.15, "poiCount": 35,
    "nodes": [ {"id":"H01","type":"start|poi|end","name":"","lat":0,"lng":0,"score":0,"atMin":0} ],
    "legs": [ {"fromIndex":0,"durationMin":18,"distanceM":9000,"throughRing":false,
               "slot":"morning","poisPassed":["P003"],"geometry":[[lat,lng]] } ]
  } ] }
```
- `atMin` = minutes since departure. `legs[i]` connects `nodes[i]` → `nodes[i+1]`.
- `geometry` may be `null`; the UI then draws a dashed straight line.
- `poisPassed` should list only POIs not already counted earlier in the route.
- The bus passes POIs without stopping; the driver never selects stops.
