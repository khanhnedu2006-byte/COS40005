# Da Nang Tour Bus – front-end (Vue 3 + Vite + Leaflet)

Front-end only. Matches the team repo: 55 nodes (50 POIs + 5 hubs), TomTom travel-time matrix in 5 slots,
central-ring ban windows for buses > 30 seats, PPO + GNN vs greedy vs shortest baselines.

## Run
```
npm install
npm run dev        # http://localhost:5173
npm run build
```
## Modes
- **Mock (default, `VITE_API_BASE` empty):** uses `public/data/od_matrix.csv` in the browser. Only the 55 known places can be searched. NOT the trained model.
- **Real:** copy `.env.example` to `.env`, set `VITE_API_BASE=/` and `VITE_API_TARGET=http://localhost:8000`. See `docs/API_CONTRACT.md`.

## Refresh static data from the team repo
```
python tools/export_static_data.py /path/to/COS40005-main
```
Never put TomTom/Goong keys in this project.
