# Data layout (not committed)

```
data/
├── raw/            # (không dùng) – log GPS gốc nằm ở ../dataset/
├── interim/gps/    # bus_XX_clean.csv, bus_XX_stops.csv, stats.json
├── processed/      # nodes.parquet, edges.parquet, edge_index.npy
├── cache/          # cache của OSMnx (Overpass API)
└── video/          # Phase 2 – video cabin, chỉ trên ổ mã hoá, KHÔNG commit
```
