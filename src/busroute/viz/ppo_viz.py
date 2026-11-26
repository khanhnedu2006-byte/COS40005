"""Trực quan hóa lộ trình PPO trên đồ thị đường rút gọn (MapLibre GL JS + PNG tĩnh).

Cung cấp hai đầu ra dùng chung dữ liệu mô phỏng ``rollout_trace``:
  * Bản đồ HTML tương tác (``build_html_visualization``): dùng MapLibre GL JS
    (tái sử dụng gói ``Web/assets/vendor/maplibre-gl.{css,js}`` có sẵn và nền
    OpenFreeMap) phát lại từng step trên bản đồ đường bộ (có khôi phục đường cong
    từ ``via_nodes``), tô màu cạnh theo bề rộng mặt đường, vẽ biểu đồ tích lũy
    reward và so sánh các chính sách.
  * Ảnh PNG 3 khung (``save_static_figure``): bản đồ tuyến đường, đường cong
    reward theo step và biểu đồ so sánh điểm POI giữa các chính sách.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from busroute.routing.ppo_env import BusRouteEnv, GraphData
from busroute.routing.ppo_train import POLICIES

ROOT = Path(__file__).resolve().parents[3]
ORIG_NODES_FILE = ROOT / "data" / "stage1" / "nodes_base.parquet"
RING_FILE = ROOT / "data" / "references" / "central_ring.geojson"
DEFAULT_BENCHMARK = ROOT / "results" / "ppo_reduced.json"
VENDOR_DIR = ROOT / "Web" / "assets" / "vendor"
MAPLIBRE_CDN = "https://unpkg.com/maplibre-gl@5.24.0/dist"


def _maplibre_asset_urls(out_html: Path) -> tuple[str, str]:
    """Chọn đường dẫn CSS/JS của MapLibre GL JS (ưu tiên gói nội bộ trong ``Web/assets/vendor``).

    Args:
        out_html: Đường dẫn tệp HTML đầu ra để tính đường dẫn tương đối.

    Returns:
        Bộ ``(css_url, js_url)`` trỏ tới bộ thư viện MapLibre cục bộ nếu ``out_html``
        nằm trong thư mục gốc dự án, ngược lại trỏ tới CDN ``unpkg.com``.
    """
    css_local = VENDOR_DIR / "maplibre-gl.css"
    js_local = VENDOR_DIR / "maplibre-gl.js"
    resolved_out = Path(out_html).resolve()
    if css_local.is_file() and js_local.is_file() and resolved_out.is_relative_to(ROOT):
        css_rel = Path(os.path.relpath(css_local, resolved_out.parent)).as_posix()
        js_rel = Path(os.path.relpath(js_local, resolved_out.parent)).as_posix()
        return css_rel, js_rel
    return f"{MAPLIBRE_CDN}/maplibre-gl.css", f"{MAPLIBRE_CDN}/maplibre-gl.js"


def load_orig_coords(nodes_path: Path = ORIG_NODES_FILE) -> np.ndarray | None:
    """Nạp tọa độ ``(lon, lat)`` của toàn bộ nút gốc để khôi phục hình học ``via_nodes``.

    Args:
        nodes_path: Đường dẫn tới ``nodes_base.parquet`` chứa ``node_idx``, ``lon``, ``lat``.

    Returns:
        Mảng ``float64`` kích thước ``(max_node_idx + 1, 2)`` chứa ``(lon, lat)``
        đánh chỉ mục theo ``node_idx``, hoặc ``None`` nếu tệp không tồn tại.
    """
    path = Path(nodes_path)
    if not path.is_file():
        return None
    df = pd.read_parquet(path, columns=["node_idx", "lon", "lat"])
    max_idx = int(df["node_idx"].max())
    coords = np.full((max_idx + 1, 2), np.nan, dtype=np.float64)
    idx = df["node_idx"].to_numpy(dtype=np.int64)
    coords[idx, 0] = df["lon"].to_numpy(dtype=np.float64)
    coords[idx, 1] = df["lat"].to_numpy(dtype=np.float64)
    return coords


def expand_edge_coords(
    graph: GraphData,
    edge_idx: int,
    orig_coords: np.ndarray | None = None,
) -> list[list[float]]:
    """Trả về chuỗi tọa độ ``[[lat, lon], ...]`` của cạnh ``edge_idx``, gồm cả ``via_nodes``.

    Args:
        graph: Dữ liệu đồ thị rút gọn ``GraphData``.
        edge_idx: Chỉ số cạnh trong mảng CSR của ``graph``.
        orig_coords: Mảng tọa độ gốc từ ``load_orig_coords()`` để chèn các nút đã gộp;
            nếu ``None`` thì chỉ nối thẳng hai đầu cạnh.

    Returns:
        Danh sách các cặp ``[lat, lon]`` theo đúng thứ tự đi qua cạnh.
    """
    u = int(graph.src[edge_idx])
    v = int(graph.dst[edge_idx])
    pts: list[list[float]] = [[float(graph.coords[u, 1]), float(graph.coords[u, 0])]]
    via_raw = graph.via_nodes[edge_idx] if hasattr(graph, "via_nodes") else ""
    if orig_coords is not None and via_raw:
        for token in via_raw.split(";"):
            if not token:
                continue
            vid = int(token)
            if 0 <= vid < len(orig_coords) and np.all(np.isfinite(orig_coords[vid])):
                pts.append([float(orig_coords[vid, 1]), float(orig_coords[vid, 0])])
    pts.append([float(graph.coords[v, 1]), float(graph.coords[v, 0])])
    return pts


def rollout_trace(
    graph: GraphData,
    policy_name: str,
    model: Any = None,
    *,
    seed: int = 0,
    start: str | int | None = None,
    end: str | int | None = None,
    start_minute: int | None = None,
    budget_s: float | None = None,
    orig_coords: np.ndarray | None = None,
    **env_kwargs: Any,
) -> dict[str, Any]:
    """Chạy một episode hoàn chỉnh và ghi lại toàn bộ nhật ký từng step để vẽ.

    Args:
        graph: Dữ liệu đồ thị rút gọn dùng chung.
        policy_name: Tên chính sách (``"ppo"``, ``"greedy"``, ``"shortest"``, ``"random"``).
        model: Mô hình ``MaskablePPO`` (bắt buộc khi ``policy_name == "ppo"``).
        seed: Seed cho ``env.reset()`` và bộ chọn ngẫu nhiên.
        start: Mã POI hoặc chỉ số nút xuất phát tùy chọn.
        end: Mã POI hoặc chỉ số nút đích tùy chọn.
        start_minute: Phút khởi hành trong ngày (0..1439) tùy chọn.
        budget_s: Ngân sách thời gian (giây) tùy chọn.
        orig_coords: Tọa độ nút gốc để bung ``via_nodes`` khi vẽ đường cong.
        **env_kwargs: Tham số bổ sung truyền vào ``BusRouteEnv``.

    Returns:
        Dictionary chứa thông tin chuyến đi, ``summary`` tổng kết và danh sách
        ``steps`` chi tiết từng cạnh đã đi qua.

    Raises:
        ValueError: Nếu ``policy_name`` không hợp lệ hoặc chọn ``"ppo"`` mà ``model`` là ``None``.
    """
    if policy_name not in POLICIES:
        raise ValueError(f"Chính sách không hợp lệ: {policy_name}")
    if policy_name == "ppo" and model is None:
        raise ValueError("Cần truyền mô hình MaskablePPO khi chạy chính sách 'ppo'")

    pick = POLICIES[policy_name]
    env = BusRouteEnv(graph, **env_kwargs)
    options: dict[str, Any] = {}
    if start is not None:
        options["start"] = start
    if end is not None:
        options["end"] = end
    if start_minute is not None:
        options["start_minute"] = start_minute
    if budget_s is not None:
        options["budget_s"] = budget_s

    obs, reset_info = env.reset(seed=seed, options=options or None)
    rng = np.random.default_rng(seed)
    steps: list[dict[str, Any]] = []
    cum_return = 0.0
    done = False
    info: dict[str, Any] = {}

    while not done:
        mask = env.action_masks()
        action = pick(obs, mask, rng, model)
        obs, reward, terminated, truncated, info = env.step(action)
        cum_return += float(reward)
        raw_coords = expand_edge_coords(graph, int(info["edge_idx"]), orig_coords)
        coords = [[round(lat, 6), round(lon, 6)] for lat, lon in raw_coords]
        steps.append(
            {
                "step": int(info["step"]),
                "from_label": str(info["from_label"]),
                "to_label": str(info["to_label"]),
                "road_name": str(info["road_name"]),
                "length_m": round(float(info["length_m"]), 2),
                "width_m": None if info["width_m"] is None else round(float(info["width_m"]), 2),
                "narrow": bool(info["narrow"]),
                "dt_s": round(float(info["dt_s"]), 2),
                "reward": round(float(reward), 4),
                "cum_return": round(cum_return, 4),
                "gained_score": round(float(info["gained_score"]), 4),
                "width_reward": round(float(info["width_reward"]), 4),
                "new_pois": list(info.get("new_pois", [])),
                "elapsed_min": round(float(info["elapsed_min"]), 2),
                "budget_min": round(float(info["budget_min"]), 2),
                "coords": coords,
            }
        )
        done = terminated or truncated

    start_node = int(reset_info["start"])
    end_node = int(reset_info["end"])
    return {
        "policy": policy_name,
        "start": start_node,
        "end": end_node,
        "start_name": str(reset_info["start_name"]),
        "end_name": str(reset_info["end_name"]),
        "start_coord": [
            round(float(graph.coords[start_node, 1]), 6),
            round(float(graph.coords[start_node, 0]), 6),
        ],
        "end_coord": [
            round(float(graph.coords[end_node, 1]), 6),
            round(float(graph.coords[end_node, 0]), 6),
        ],
        "summary": {
            "steps": len(steps),
            "score": float(info.get("score", 0.0)),
            "pois": int(info.get("pois", 0)),
            "return": float(cum_return),
            "elapsed_min": float(info.get("elapsed_min", 0.0)),
            "budget_min": float(info.get("budget_min", reset_info["budget_min"])),
            "narrow_m": float(info.get("narrow_m", 0.0)),
            "unknown_width_m": float(info.get("unknown_width_m", 0.0)),
            "success": bool(info.get("success", False)),
        },
        "steps": steps,
    }


def _extract_ring_coords(ring_geojson: dict[str, Any] | None) -> list[list[list[float]]]:
    """Trích xuất danh sách vòng tọa độ ``[[lat, lon], ...]`` từ GeoJSON vành đai."""
    if not ring_geojson:
        return []
    rings: list[list[list[float]]] = []
    for feat in ring_geojson.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        if gtype == "Polygon":
            for ring in coords:
                rings.append([[round(float(lat), 6), round(float(lon), 6)] for lon, lat in ring])
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    rings.append([[round(float(lat), 6), round(float(lon), 6)] for lon, lat in ring])
    return rings


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PPO Reduced Road Graph — MapLibre GL Replay</title>
<link rel="stylesheet" href="__MAPLIBRE_CSS__">
<style>
  :root {
    --bg: #0f172a;
    --panel: #111827;
    --card: #1e293b;
    --border: #334155;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --wide: #10b981;
    --narrow: #ef4444;
    --unknown: #f59e0b;
    --accent: #38bdf8;
    --gold: #fbbf24;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    overflow: hidden;
  }
  .layout {
    display: grid;
    grid-template-columns: 1.7fr 1fr;
    height: 100vh;
  }
  .map-wrap {
    position: relative;
    height: 100%;
    background: #0b1220;
  }
  #map, #fallback-canvas {
    width: 100%;
    height: 100%;
    display: block;
  }
  #fallback-canvas { display: none; background: #0b1220; }
  .maplibregl-popup-content {
    background: rgba(15, 23, 42, 0.95);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 12px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
  }
  .maplibregl-popup-tip { border-top-color: rgba(15, 23, 42, 0.95) !important; }
  .map-overlay {
    position: absolute;
    top: 12px;
    left: 12px;
    z-index: 20;
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 12px;
    backdrop-filter: blur(6px);
    min-width: 178px;
  }
  .legend-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
  .swatch { width: 16px; height: 4px; border-radius: 2px; display: inline-block; }
  .dot-swatch { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .map-ctrl {
    margin-top: 6px;
    width: 100%;
    background: var(--card);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 8px;
    cursor: pointer;
    font-size: 12px;
  }
  .map-ctrl:hover { border-color: var(--accent); }
  .follow-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 7px;
    color: var(--muted);
    cursor: pointer;
    user-select: none;
  }
  .sidebar {
    background: var(--panel);
    border-left: 1px solid var(--border);
    padding: 14px 16px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .tabs { display: flex; gap: 6px; flex-wrap: wrap; }
  .tab {
    background: var(--card);
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    text-transform: uppercase;
  }
  .tab.active {
    background: #0284c7;
    color: #fff;
    border-color: var(--accent);
  }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px;
  }
  .trip-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 8px;
  }
  .trip-title { font-size: 15px; font-weight: 700; line-height: 1.3; }
  .pill {
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 999px;
    white-space: nowrap;
  }
  .pill.ok { background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
  .pill.fail { background: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
  .kpis {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-top: 10px;
  }
  .kpi { background: rgba(15, 23, 42, 0.55); border-radius: 8px; padding: 8px; }
  .kpi-label { font-size: 11px; color: var(--muted); }
  .kpi-val { font-size: 14px; font-weight: 700; margin-top: 2px; }
  .controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .btn {
    background: #0f172a;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 6px 10px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
  }
  .btn.primary { background: #0284c7; border-color: var(--accent); }
  .btn:hover { border-color: var(--accent); }
  select.speed {
    background: #0f172a;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 5px 8px;
    font-size: 12px;
  }
  .slider-row { display: flex; align-items: center; gap: 10px; margin-top: 8px; }
  input[type=range] { flex: 1; accent-color: var(--accent); }
  .step-line { font-size: 13px; margin: 5px 0; }
  .badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 700;
  }
  .badge.wide { background: rgba(16, 185, 129, 0.2); color: #34d399; }
  .badge.narrow { background: rgba(239, 68, 68, 0.2); color: #f87171; }
  .badge.unknown { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
  .prog-track {
    height: 7px;
    background: #0f172a;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 6px;
  }
  .prog-fill { height: 100%; background: var(--accent); width: 0%; transition: width 0.08s; }
  h3 { margin: 0 0 8px 0; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  svg.chart { width: 100%; height: 130px; display: block; cursor: pointer; }
  .poi-list { display: flex; flex-wrap: wrap; gap: 6px; max-height: 110px; overflow-y: auto; }
  .poi-chip {
    font-size: 11px;
    background: rgba(251, 191, 36, 0.14);
    color: var(--gold);
    border: 1px solid rgba(251, 191, 36, 0.35);
    border-radius: 999px;
    padding: 3px 8px;
  }
  .bench-row { margin: 6px 0; font-size: 12px; }
  .bench-head { display: flex; justify-content: space-between; margin-bottom: 3px; }
  .bench-bar { height: 6px; background: #0f172a; border-radius: 4px; overflow: hidden; }
  .bench-fill { height: 100%; background: var(--accent); }
  @media (max-width: 960px) {
    body { overflow: auto; height: auto; }
    .layout { grid-template-columns: 1fr; grid-template-rows: 55vh auto; height: auto; }
  }
</style>
</head>
<body>
<div class="layout">
  <div class="map-wrap">
    <div id="map"></div>
    <canvas id="fallback-canvas"></canvas>
    <div class="map-overlay">
      <div style="font-weight:700;margin-bottom:4px;">Road Width &amp; POIs</div>
      <div class="legend-row"><span class="swatch" style="background:var(--wide)"></span> Wide (&ge; 5.5 m)</div>
      <div class="legend-row"><span class="swatch" style="background:var(--narrow)"></span> Narrow (&lt; 5.5 m)</div>
      <div class="legend-row"><span class="swatch" style="background:var(--unknown)"></span> Unknown width</div>
      <div class="legend-row"><span class="swatch" style="border-top:2px dashed #fb923c;height:0"></span> Central ring</div>
      <div class="legend-row"><span class="dot-swatch" style="background:#22c55e"></span> Start POI (A)</div>
      <div class="legend-row"><span class="dot-swatch" style="background:#ef4444"></span> End POI (B)</div>
      <div class="legend-row"><span class="dot-swatch" style="background:var(--gold)"></span> Visited POI</div>
      <select class="map-ctrl" id="basemap-sel" title="Basemap style">
        <option value="bright" selected>Map: OpenFreeMap Bright</option>
        <option value="positron">Map: OpenFreeMap Positron</option>
        <option value="dark">Map: Offline Dark</option>
      </select>
      <label class="follow-row"><input type="checkbox" id="follow-bus"> Follow bus</label>
      <button class="map-ctrl" id="fit-btn">Fit Route</button>
    </div>
  </div>
  <div class="sidebar">
    <div class="tabs" id="policy-tabs"></div>

    <div class="card">
      <div class="trip-head">
        <div class="trip-title" id="trip-title">—</div>
        <span class="pill" id="trip-status">—</span>
      </div>
      <div class="kpis">
        <div class="kpi"><div class="kpi-label">POI Score</div><div class="kpi-val" id="kpi-score">0</div></div>
        <div class="kpi"><div class="kpi-label">POIs Collected</div><div class="kpi-val" id="kpi-pois">0</div></div>
        <div class="kpi"><div class="kpi-label">Time / Budget</div><div class="kpi-val" id="kpi-time">0 / 0m</div></div>
        <div class="kpi"><div class="kpi-label">Narrow Road</div><div class="kpi-val" id="kpi-narrow">0 m</div></div>
        <div class="kpi"><div class="kpi-label">Cum. Return</div><div class="kpi-val" id="kpi-return">0.00</div></div>
        <div class="kpi"><div class="kpi-label">Total Steps</div><div class="kpi-val" id="kpi-steps">0</div></div>
      </div>
    </div>

    <div class="card">
      <h3>Step Playback</h3>
      <div class="controls">
        <button class="btn primary" id="play-btn">Play</button>
        <button class="btn" id="prev-btn">Prev</button>
        <button class="btn" id="next-btn">Next</button>
        <button class="btn" id="end-btn">End</button>
        <select class="speed" id="speed-sel">
          <option value="1">1x</option>
          <option value="3" selected>3x</option>
          <option value="8">8x</option>
          <option value="20">20x</option>
        </select>
      </div>
      <div class="slider-row">
        <input type="range" id="step-slider" min="0" max="0" value="0">
        <span id="step-counter" style="font-size:12px;color:var(--muted);min-width:68px;text-align:right;">0 / 0</span>
      </div>
    </div>

    <div class="card">
      <h3>Current Step</h3>
      <div class="step-line" id="step-nodes">—</div>
      <div class="step-line" id="step-road">—</div>
      <div class="step-line" id="step-reward">—</div>
      <div class="prog-track"><div class="prog-fill" id="budget-bar"></div></div>
    </div>

    <div class="card">
      <h3>Cumulative Return Across Steps (click to seek)</h3>
      <svg class="chart" id="return-chart" viewBox="0 0 360 130" preserveAspectRatio="none"></svg>
    </div>

    <div class="card">
      <h3>Collected POIs Up To Step</h3>
      <div class="poi-list" id="poi-list"></div>
    </div>

    <div class="card" id="bench-card" style="display:none;">
      <h3>Evaluation Benchmark</h3>
      <div id="bench-rows"></div>
    </div>
  </div>
</div>

<script id="ppo-payload" type="application/json">__PAYLOAD__</script>
<script src="__MAPLIBRE_JS__"></script>
<script>
(() => {
  const DATA = JSON.parse(document.getElementById("ppo-payload").textContent);
  const policyNames = Object.keys(DATA.traces);
  let activePolicy = policyNames.includes("ppo") ? "ppo" : policyNames[0];
  let curStep = DATA.traces[activePolicy].steps.length;
  let playing = false;
  let timerId = null;

  const COLOR = { wide: "#10b981", narrow: "#ef4444", unknown: "#f59e0b" };
  const toLngLat = ([lat, lon]) => [lon, lat];
  const EMPTY = () => ({ type: "FeatureCollection", features: [] });
  function edgeKind(s) {
    if (s.narrow) return "narrow";
    if (s.width_m === null || s.width_m === undefined) return "unknown";
    return "wide";
  }

  const STYLES = {
    bright: "https://tiles.openfreemap.org/styles/bright",
    positron: "https://tiles.openfreemap.org/styles/positron",
    dark: {
      version: 8,
      sources: {},
      layers: [{ id: "bg", type: "background", paint: { "background-color": "#0b1220" } }],
    },
  };

  function traceBounds(tr) {
    const pts = [tr.start_coord, tr.end_coord];
    tr.steps.forEach(s => s.coords.forEach(c => pts.push(c)));
    let minLat = 90, maxLat = -90, minLon = 180, maxLon = -180;
    pts.forEach(([la, lo]) => {
      if (la < minLat) minLat = la; if (la > maxLat) maxLat = la;
      if (lo < minLon) minLon = lo; if (lo > maxLon) maxLon = lo;
    });
    return [[minLon, minLat], [maxLon, maxLat]];
  }

  function stepMetrics() {
    const tr = DATA.traces[activePolicy];
    let narrowM = 0, poiScore = 0;
    const visited = [tr.start_name];
    const visitedSet = new Set(visited);
    for (let i = 0; i < curStep; i++) {
      const s = tr.steps[i];
      if (s.narrow) narrowM += s.length_m;
      poiScore += s.gained_score;
      (s.new_pois || []).forEach(p => {
        if (!visitedSet.has(p)) { visitedSet.add(p); visited.push(p); }
      });
    }
    return { tr, narrowM, poiScore, visited, visitedSet };
  }

  // ---- MapLibre GL JS backend (with HTML5 Canvas fallback if WebGL is unavailable) ----
  let useMapLibre = typeof maplibregl !== "undefined";
  let map = null;
  let styleLoaded = false;
  let fellBackToDark = false;
  let hoverPopup = null;

  function ringGeoJson() {
    return {
      type: "FeatureCollection",
      features: (DATA.ring || []).map(ring => ({
        type: "Feature",
        properties: {},
        geometry: { type: "Polygon", coordinates: [ring.map(toLngLat)] },
      })),
    };
  }

  function previewGeoJson(tr) {
    return {
      type: "FeatureCollection",
      features: tr.steps.map(s => ({
        type: "Feature",
        properties: { step: s.step },
        geometry: { type: "LineString", coordinates: s.coords.map(toLngLat) },
      })),
    };
  }

  function progressGeoJson(tr) {
    return {
      type: "FeatureCollection",
      features: tr.steps.slice(0, curStep).map(s => ({
        type: "Feature",
        properties: {
          step: s.step,
          kind: edgeKind(s),
          road_name: s.road_name,
          length_m: s.length_m,
          width_m: s.width_m,
          reward: s.reward,
        },
        geometry: { type: "LineString", coordinates: s.coords.map(toLngLat) },
      })),
    };
  }

  function poisGeoJson(tr, visitedSet) {
    return {
      type: "FeatureCollection",
      features: (DATA.pois || []).map(p => {
        let role = "unvisited";
        if (p.name === tr.start_name) role = "start";
        else if (p.name === tr.end_name) role = "end";
        else if (visitedSet.has(p.name)) role = "visited";
        return {
          type: "Feature",
          properties: { id: p.id, name: p.name, score: p.score, role },
          geometry: { type: "Point", coordinates: [p.lon, p.lat] },
        };
      }),
    };
  }

  function busGeoJson(tr) {
    const pos = curStep > 0 ? tr.steps[curStep - 1].coords.at(-1) : tr.start_coord;
    return {
      type: "FeatureCollection",
      features: [{
        type: "Feature",
        properties: {},
        geometry: { type: "Point", coordinates: toLngLat(pos) },
      }],
    };
  }

  function fitActiveRoute() {
    const tr = DATA.traces[activePolicy];
    if (useMapLibre && map) {
      map.fitBounds(traceBounds(tr), { padding: 48, duration: 350 });
    } else {
      drawCanvas(tr);
    }
  }

  function updateMapData(shouldFit) {
    if (!useMapLibre || !map || !styleLoaded) return;
    const { tr, visitedSet } = stepMetrics();
    map.getSource("route-preview").setData(previewGeoJson(tr));
    map.getSource("route-progress").setData(progressGeoJson(tr));
    map.getSource("pois").setData(poisGeoJson(tr, visitedSet));
    map.getSource("bus-marker").setData(busGeoJson(tr));
    if (shouldFit) {
      map.fitBounds(traceBounds(tr), { padding: 48, duration: 0 });
    } else if (document.getElementById("follow-bus").checked && curStep > 0) {
      const pos = tr.steps[curStep - 1].coords.at(-1);
      map.easeTo({ center: toLngLat(pos), duration: 150 });
    }
  }

  function addDataLayers() {
    styleLoaded = true;
    map.addSource("central-ring", { type: "geojson", data: ringGeoJson() });
    map.addLayer({
      id: "central-ring-fill",
      type: "fill",
      source: "central-ring",
      paint: { "fill-color": "#fb923c", "fill-opacity": 0.04 },
    });
    map.addLayer({
      id: "central-ring-line",
      type: "line",
      source: "central-ring",
      paint: {
        "line-color": "#fb923c",
        "line-width": 2,
        "line-dasharray": [4, 3],
        "line-opacity": 0.85,
      },
    });

    map.addSource("route-preview", { type: "geojson", data: EMPTY() });
    map.addLayer({
      id: "route-preview-line",
      type: "line",
      source: "route-preview",
      paint: {
        "line-color": "#64748b",
        "line-width": 2.5,
        "line-dasharray": [3, 3],
        "line-opacity": 0.45,
      },
    });

    map.addSource("route-progress", { type: "geojson", data: EMPTY() });
    map.addLayer({
      id: "route-progress-line",
      type: "line",
      source: "route-progress",
      layout: { "line-cap": "round", "line-join": "round" },
      paint: {
        "line-color": ["match", ["get", "kind"], "wide", "#10b981", "narrow", "#ef4444", "#f59e0b"],
        "line-width": 5,
        "line-opacity": 0.94,
      },
    });

    map.addSource("pois", { type: "geojson", data: EMPTY() });
    map.addLayer({
      id: "pois-circle",
      type: "circle",
      source: "pois",
      paint: {
        "circle-radius": ["match", ["get", "role"], "start", 8, "end", 8, "visited", 6.5, 4.5],
        "circle-color": ["match", ["get", "role"], "start", "#22c55e", "end", "#ef4444", "visited", "#fbbf24", "#94a3b8"],
        "circle-stroke-width": ["match", ["get", "role"], "unvisited", 1, 2],
        "circle-stroke-color": ["match", ["get", "role"], "unvisited", "#334155", "#ffffff"],
        "circle-opacity": ["match", ["get", "role"], "unvisited", 0.78, 1.0],
      },
    });

    map.addSource("bus-marker", { type: "geojson", data: EMPTY() });
    map.addLayer({
      id: "bus-halo",
      type: "circle",
      source: "bus-marker",
      paint: { "circle-radius": 13, "circle-color": "#38bdf8", "circle-opacity": 0.25 },
    });
    map.addLayer({
      id: "bus-dot",
      type: "circle",
      source: "bus-marker",
      paint: {
        "circle-radius": 6.5,
        "circle-color": "#38bdf8",
        "circle-stroke-width": 2.5,
        "circle-stroke-color": "#ffffff",
      },
    });

    updateMapData(true);
  }

  function bindPopups() {
    hoverPopup = new maplibregl.Popup({ closeButton: false, closeOnClick: false });
    map.on("mousemove", "pois-circle", ev => {
      map.getCanvas().style.cursor = "pointer";
      const f = ev.features[0];
      const p = f.properties;
      hoverPopup
        .setLngLat(f.geometry.coordinates)
        .setHTML(`<strong>${p.id}: ${p.name}</strong><br>Score: ${Number(p.score).toFixed(2)}`)
        .addTo(map);
    });
    map.on("mouseleave", "pois-circle", () => {
      map.getCanvas().style.cursor = "";
      hoverPopup.remove();
    });
    map.on("mousemove", "route-progress-line", ev => {
      map.getCanvas().style.cursor = "pointer";
      const p = ev.features[0].properties;
      const wTxt = p.width_m !== null && p.width_m !== undefined ? `${Number(p.width_m).toFixed(1)} m` : "unknown";
      hoverPopup
        .setLngLat(ev.lngLat)
        .setHTML(`<strong>#${p.step}: ${p.road_name}</strong><br>${Number(p.length_m).toFixed(0)} m | w=${wTxt} | r=${Number(p.reward).toFixed(3)}`)
        .addTo(map);
    });
    map.on("mouseleave", "route-progress-line", () => {
      map.getCanvas().style.cursor = "";
      hoverPopup.remove();
    });
  }

  function enableCanvasFallback() {
    useMapLibre = false;
    document.getElementById("map").style.display = "none";
    document.getElementById("fallback-canvas").style.display = "block";
    document.getElementById("basemap-sel").style.display = "none";
  }

  if (useMapLibre) {
    try {
      const tr0 = DATA.traces[activePolicy];
      map = new maplibregl.Map({
        container: "map",
        style: STYLES.bright,
        center: toLngLat(tr0.start_coord),
        zoom: 12,
        attributionControl: true,
      });
      map.addControl(new maplibregl.NavigationControl(), "top-right");
      map.on("style.load", addDataLayers);
      map.on("load", bindPopups);
      map.on("error", () => {
        if (!styleLoaded && !fellBackToDark) {
          fellBackToDark = true;
          document.getElementById("basemap-sel").value = "dark";
          map.setStyle(STYLES.dark);
        }
      });
    } catch (_err) {
      enableCanvasFallback();
    }
  } else {
    enableCanvasFallback();
  }

  function drawCanvas(tr) {
    const canvas = document.getElementById("fallback-canvas");
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(300, rect.width);
    canvas.height = Math.max(300, rect.height);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const [[minLon, minLat], [maxLon, maxLat]] = traceBounds(tr);
    const pad = 40;
    const spanLon = Math.max(1e-5, maxLon - minLon);
    const spanLat = Math.max(1e-5, maxLat - minLat);
    const scale = Math.min((canvas.width - 2 * pad) / spanLon, (canvas.height - 2 * pad) / spanLat);
    const ox = (canvas.width - spanLon * scale) / 2;
    const oy = (canvas.height - spanLat * scale) / 2;
    const proj = ([la, lo]) => [ox + (lo - minLon) * scale, canvas.height - oy - (la - minLat) * scale];

    ctx.strokeStyle = "rgba(148,163,184,0.35)";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    tr.steps.forEach(s => {
      ctx.beginPath();
      s.coords.forEach((c, idx) => {
        const [x, y] = proj(c);
        idx === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
    ctx.setLineDash([]);

    const visited = new Set([tr.start_name]);
    for (let i = 0; i < curStep; i++) {
      const s = tr.steps[i];
      (s.new_pois || []).forEach(n => visited.add(n));
      ctx.strokeStyle = COLOR[edgeKind(s)];
      ctx.lineWidth = 4;
      ctx.beginPath();
      s.coords.forEach((c, idx) => {
        const [x, y] = proj(c);
        idx === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    (DATA.pois || []).forEach(p => {
      const [x, y] = proj([p.lat, p.lon]);
      const isStart = p.name === tr.start_name;
      const isEnd = p.name === tr.end_name;
      const isVis = visited.has(p.name);
      ctx.beginPath();
      ctx.arc(x, y, isStart || isEnd ? 7 : (isVis ? 6 : 3.5), 0, Math.PI * 2);
      ctx.fillStyle = isStart ? "#22c55e" : (isEnd ? "#ef4444" : (isVis ? "#fbbf24" : "#64748b"));
      ctx.fill();
    });

    const curCoord = curStep === 0 ? tr.start_coord : tr.steps[curStep - 1].coords.at(-1);
    const [bx, by] = proj(curCoord);
    ctx.beginPath();
    ctx.arc(bx, by, 7, 0, Math.PI * 2);
    ctx.fillStyle = "#38bdf8";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
  }

  function renderTabs() {
    const el = document.getElementById("policy-tabs");
    el.innerHTML = "";
    policyNames.forEach(name => {
      const b = document.createElement("button");
      b.className = "tab" + (name === activePolicy ? " active" : "");
      b.textContent = name;
      b.onclick = () => {
        stopPlay();
        activePolicy = name;
        const maxSteps = DATA.traces[activePolicy].steps.length;
        curStep = Math.min(curStep, maxSteps);
        document.getElementById("step-slider").max = maxSteps;
        renderTabs();
        renderStep();
        fitActiveRoute();
      };
      el.appendChild(b);
    });
  }

  function renderChart(tr) {
    const svg = document.getElementById("return-chart");
    const W = 360, H = 130, pad = 12;
    const vals = [0, ...tr.steps.map(s => s.cum_return)];
    let minV = Math.min(...vals), maxV = Math.max(...vals);
    if (Math.abs(maxV - minV) < 1e-6) { minV -= 1; maxV += 1; }
    const n = Math.max(1, tr.steps.length);
    const sx = i => pad + (i / n) * (W - 2 * pad);
    const sy = v => H - pad - ((v - minV) / (maxV - minV)) * (H - 2 * pad);

    const pts = vals.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(" ");
    const zeroY = sy(Math.min(maxV, Math.max(minV, 0))).toFixed(1);
    let dots = "";
    tr.steps.forEach((s, i) => {
      if (s.gained_score > 0) {
        dots += `<circle cx="${sx(i + 1).toFixed(1)}" cy="${sy(s.cum_return).toFixed(1)}" r="3.2" fill="#fbbf24" />`;
      }
    });
    const cx = sx(curStep).toFixed(1);
    const cy = sy(vals[curStep]).toFixed(1);
    svg.innerHTML = `
      <line x1="${pad}" y1="${zeroY}" x2="${W - pad}" y2="${zeroY}" stroke="#334155" stroke-dasharray="3 3"/>
      <polyline fill="none" stroke="#38bdf8" stroke-width="2" points="${pts}"/>
      ${dots}
      <line x1="${cx}" y1="${pad}" x2="${cx}" y2="${H - pad}" stroke="#f8fafc" stroke-width="1" stroke-dasharray="2 2"/>
      <circle cx="${cx}" cy="${cy}" r="4" fill="#ffffff" stroke="#38bdf8" stroke-width="2"/>
    `;
    svg.onclick = ev => {
      const r = svg.getBoundingClientRect();
      const frac = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width));
      curStep = Math.round(frac * tr.steps.length);
      renderStep();
    };
  }

  function renderBenchmark() {
    const rows = DATA.benchmark || [];
    const card = document.getElementById("bench-card");
    const box = document.getElementById("bench-rows");
    if (!rows.length) { card.style.display = "none"; return; }
    card.style.display = "block";
    const maxScore = Math.max(1, ...rows.map(r => r.mean_score || 0));
    box.innerHTML = rows.map(r => {
      const pct = Math.max(2, ((r.mean_score || 0) / maxScore) * 100);
      const narrow = r.mean_narrow_m !== undefined ? ` | narrow ${r.mean_narrow_m.toFixed(0)}m` : "";
      const succ = r.success_rate !== undefined ? ` | ${(r.success_rate * 100).toFixed(0)}% ok` : "";
      return `
        <div class="bench-row">
          <div class="bench-head">
            <strong>${r.policy.toUpperCase()}</strong>
            <span>score ${(r.mean_score || 0).toFixed(1)} (${(r.mean_pois || 0).toFixed(1)} POIs${narrow}${succ})</span>
          </div>
          <div class="bench-bar"><div class="bench-fill" style="width:${pct.toFixed(1)}%"></div></div>
        </div>`;
    }).join("");
  }

  function renderStep() {
    const { tr, narrowM, poiScore, visited, visitedSet } = stepMetrics();
    const sum = tr.summary;
    const total = tr.steps.length;
    const slider = document.getElementById("step-slider");
    slider.max = total;
    slider.value = curStep;
    document.getElementById("step-counter").textContent = `${curStep} / ${total}`;
    document.getElementById("trip-title").textContent = `${tr.start_name} → ${tr.end_name}`;
    const pill = document.getElementById("trip-status");
    pill.textContent = sum.success ? "Reached B" : "Failed";
    pill.className = "pill " + (sum.success ? "ok" : "fail");

    const curObj = curStep > 0 ? tr.steps[curStep - 1] : null;
    const elapsed = curObj ? curObj.elapsed_min : 0;
    const cumRet = curObj ? curObj.cum_return : 0;

    document.getElementById("kpi-score").textContent = poiScore.toFixed(2);
    document.getElementById("kpi-pois").textContent = `${visitedSet.size} / ${sum.pois || visitedSet.size}`;
    document.getElementById("kpi-time").textContent = `${elapsed.toFixed(1)} / ${sum.budget_min.toFixed(1)}m`;
    document.getElementById("kpi-narrow").textContent = `${narrowM.toFixed(0)} m`;
    document.getElementById("kpi-return").textContent = `${cumRet >= 0 ? "+" : ""}${cumRet.toFixed(2)}`;
    document.getElementById("kpi-steps").textContent = `${total}`;
    document.getElementById("budget-bar").style.width = `${Math.min(100, (elapsed / Math.max(0.1, sum.budget_min)) * 100).toFixed(1)}%`;

    if (curObj) {
      const kind = edgeKind(curObj);
      const wLabel = curObj.width_m !== null ? `${curObj.width_m.toFixed(1)} m` : "unknown";
      document.getElementById("step-nodes").textContent = `#${curObj.step}: ${curObj.from_label} → ${curObj.to_label}`;
      document.getElementById("step-road").innerHTML =
        `Road: <strong>${curObj.road_name}</strong> (${curObj.length_m.toFixed(0)} m, ${curObj.dt_s.toFixed(1)} s) ` +
        `<span class="badge ${kind}">${kind.toUpperCase()} ${wLabel}</span>`;
      document.getElementById("step-reward").textContent =
        `Reward: ${curObj.reward >= 0 ? "+" : ""}${curObj.reward.toFixed(3)} ` +
        `(POI ${curObj.gained_score >= 0 ? "+" : ""}${curObj.gained_score.toFixed(2)}, width ${curObj.width_reward >= 0 ? "+" : ""}${curObj.width_reward.toFixed(3)})`;
    } else {
      document.getElementById("step-nodes").textContent = `Start at ${tr.start_name}`;
      document.getElementById("step-road").textContent = "Press Play or drag the slider to replay step-by-step.";
      document.getElementById("step-reward").textContent = "Reward: 0.000";
    }

    document.getElementById("poi-list").innerHTML = visited
      .map(name => `<span class="poi-chip">${name}</span>`)
      .join("");

    if (useMapLibre) {
      updateMapData(false);
    } else {
      drawCanvas(tr);
    }
    renderChart(tr);
  }

  function stopPlay() {
    playing = false;
    document.getElementById("play-btn").textContent = "Play";
    if (timerId !== null) { clearInterval(timerId); timerId = null; }
  }

  function startPlay() {
    const tr = DATA.traces[activePolicy];
    if (curStep >= tr.steps.length) curStep = 0;
    playing = true;
    document.getElementById("play-btn").textContent = "Pause";
    const mult = Number(document.getElementById("speed-sel").value) || 3;
    timerId = setInterval(() => {
      if (curStep >= tr.steps.length) { stopPlay(); return; }
      curStep += 1;
      renderStep();
    }, Math.max(25, Math.round(320 / mult)));
  }

  document.getElementById("play-btn").onclick = () => playing ? stopPlay() : startPlay();
  document.getElementById("prev-btn").onclick = () => { stopPlay(); curStep = Math.max(0, curStep - 1); renderStep(); };
  document.getElementById("next-btn").onclick = () => {
    stopPlay();
    curStep = Math.min(DATA.traces[activePolicy].steps.length, curStep + 1);
    renderStep();
  };
  document.getElementById("end-btn").onclick = () => {
    stopPlay();
    curStep = DATA.traces[activePolicy].steps.length;
    renderStep();
  };
  document.getElementById("speed-sel").onchange = () => { if (playing) { stopPlay(); startPlay(); } };
  document.getElementById("step-slider").oninput = ev => { stopPlay(); curStep = Number(ev.target.value); renderStep(); };
  document.getElementById("fit-btn").onclick = fitActiveRoute;
  document.getElementById("basemap-sel").onchange = ev => {
    if (!useMapLibre || !map) return;
    styleLoaded = false;
    map.setStyle(STYLES[ev.target.value] || STYLES.bright);
  };

  renderTabs();
  renderBenchmark();
  renderStep();
})();
</script>
</body>
</html>
"""


def build_html_visualization(
    traces: dict[str, dict[str, Any]],
    graph: GraphData,
    out_html: Path,
    *,
    benchmark: list[dict[str, Any]] | None = None,
    ring_geojson: dict[str, Any] | None = None,
) -> Path:
    """Tạo tệp HTML MapLibre GL JS phát lại từng step, tô màu bề rộng đường và so sánh chính sách.

    Args:
        traces: Ánh xạ ``policy_name -> trace`` trả về từ ``rollout_trace()``.
        graph: Dữ liệu đồ thị rút gọn chứa danh sách và tọa độ các POI.
        out_html: Đường dẫn tệp ``.html`` đầu ra.
        benchmark: Danh sách kết quả đánh giá từ ``results/ppo_reduced.json`` (nếu có).
        ring_geojson: Nội dung GeoJSON vành đai trung tâm (nếu có).

    Returns:
        Đường dẫn ``Path`` tới tệp HTML đã ghi.
    """
    pois = [
        {
            "id": graph.poi_ids[i],
            "name": graph.poi_names[i],
            "score": round(float(graph.poi_score[i]), 3),
            "lat": round(float(graph.coords[int(graph.poi_node[i]), 1]), 6),
            "lon": round(float(graph.coords[int(graph.poi_node[i]), 0]), 6),
            "node": int(graph.poi_node[i]),
        }
        for i in range(len(graph.poi_node))
    ]
    payload = {
        "traces": traces,
        "pois": pois,
        "ring": _extract_ring_coords(ring_geojson),
        "benchmark": benchmark or [],
    }
    safe_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    out = Path(out_html)
    css_url, js_url = _maplibre_asset_urls(out)
    html = (
        _HTML_TEMPLATE.replace("__MAPLIBRE_CSS__", css_url)
        .replace("__MAPLIBRE_JS__", js_url)
        .replace("__PAYLOAD__", safe_json)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def save_static_figure(
    traces: dict[str, dict[str, Any]],
    graph: GraphData,
    out_png: Path,
    *,
    benchmark: list[dict[str, Any]] | None = None,
    ring_geojson: dict[str, Any] | None = None,
) -> Path:
    """Xuất ảnh PNG 3 khung: bản đồ bề rộng đường, tích lũy reward và bảng so sánh.

    Args:
        traces: Ánh xạ ``policy_name -> trace`` trả về từ ``rollout_trace()``.
        graph: Dữ liệu đồ thị rút gọn dùng để vẽ nền POI.
        out_png: Đường dẫn tệp ``.png`` đầu ra.
        benchmark: Danh sách kết quả đánh giá tổng hợp (nếu ``None`` sẽ dùng trực tiếp ``traces``).
        ring_geojson: Nội dung GeoJSON vành đai trung tâm (nếu có).

    Returns:
        Đường dẫn ``Path`` tới tệp PNG đã lưu.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    primary_name = "ppo" if "ppo" in traces else next(iter(traces))
    primary = traces[primary_name]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    ax_map, ax_ret, ax_bar = axes

    # --- Khung 1: Bản đồ lộ trình theo bề rộng đường ---
    for ring in _extract_ring_coords(ring_geojson):
        lats = [p[0] for p in ring]
        lons = [p[1] for p in ring]
        ax_map.plot(lons, lats, color="#fb923c", linestyle="--", linewidth=1.1, alpha=0.8, label="Central ring")

    poi_lons = [float(graph.coords[int(n), 0]) for n in graph.poi_node]
    poi_lats = [float(graph.coords[int(n), 1]) for n in graph.poi_node]
    ax_map.scatter(poi_lons, poi_lats, s=16, color="#94a3b8", alpha=0.75, zorder=2, label="Unvisited POIs")

    if "greedy" in traces and primary_name != "greedy":
        g_segs = [
            [(pt[1], pt[0]) for pt in s["coords"]]
            for s in traces["greedy"]["steps"]
        ]
        if g_segs:
            ax_map.add_collection(
                LineCollection(g_segs, colors="#38bdf8", linewidths=1.5, linestyles=":", alpha=0.7, zorder=3, label="Greedy route")
            )

    segs_by_kind: dict[str, list[list[tuple[float, float]]]] = {"wide": [], "narrow": [], "unknown": []}
    collected_names = {primary["start_name"]}
    for s in primary["steps"]:
        kind = "narrow" if s["narrow"] else ("unknown" if s["width_m"] is None else "wide")
        segs_by_kind[kind].append([(pt[1], pt[0]) for pt in s["coords"]])
        collected_names.update(s.get("new_pois", []))

    for kind, color, label in (
        ("wide", "#10b981", "Wide (>= 5.5 m)"),
        ("narrow", "#ef4444", "Narrow (< 5.5 m)"),
        ("unknown", "#f59e0b", "Unknown width"),
    ):
        if segs_by_kind[kind]:
            ax_map.add_collection(
                LineCollection(segs_by_kind[kind], colors=color, linewidths=2.6, zorder=4, label=label)
            )

    vis_lons, vis_lats = [], []
    for i, node in enumerate(graph.poi_node):
        if graph.poi_names[i] in collected_names:
            vis_lons.append(float(graph.coords[int(node), 0]))
            vis_lats.append(float(graph.coords[int(node), 1]))
    if vis_lons:
        ax_map.scatter(vis_lons, vis_lats, s=42, color="#fbbf24", edgecolors="#1e293b", zorder=5, label="Collected POIs")

    ax_map.scatter([primary["start_coord"][1]], [primary["start_coord"][0]], s=95, marker="s", color="#22c55e", edgecolors="#0f172a", zorder=6, label="Start A")
    ax_map.scatter([primary["end_coord"][1]], [primary["end_coord"][0]], s=130, marker="*", color="#ef4444", edgecolors="#0f172a", zorder=6, label="End B")

    all_pts = [primary["start_coord"], primary["end_coord"]] + [pt for s in primary["steps"] for pt in s["coords"]]
    all_lats = [p[0] for p in all_pts]
    all_lons = [p[1] for p in all_pts]
    dlat = max(0.008, (max(all_lats) - min(all_lats)) * 0.12)
    dlon = max(0.008, (max(all_lons) - min(all_lons)) * 0.12)
    ax_map.set_xlim(min(all_lons) - dlon, max(all_lons) + dlon)
    ax_map.set_ylim(min(all_lats) - dlat, max(all_lats) + dlat)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_title(f"{primary_name.upper()}: {primary['start_name']} → {primary['end_name']}", fontsize=10, fontweight="bold")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    handles, labels = ax_map.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax_map.legend(dedup.values(), dedup.keys(), fontsize=7.5, loc="best", framealpha=0.9)

    # --- Khung 2: Đường cong tích lũy reward qua các step ---
    palette = {"ppo": "#2563eb", "greedy": "#10b981", "shortest": "#64748b", "random": "#f59e0b"}
    for pname, tr in traces.items():
        ys = [0.0] + [s["cum_return"] for s in tr["steps"]]
        xs = list(range(len(ys)))
        ax_ret.plot(xs, ys, label=f"{pname.upper()} ({ys[-1]:+.1f})", color=palette.get(pname, "#0f172a"), linewidth=2.0)
        if pname == primary_name:
            poi_x = [s["step"] for s in tr["steps"] if s["gained_score"] > 0]
            poi_y = [s["cum_return"] for s in tr["steps"] if s["gained_score"] > 0]
            if poi_x:
                ax_ret.scatter(poi_x, poi_y, color="#fbbf24", edgecolors="#1e293b", s=28, zorder=5, label="POI collected")
    ax_ret.axhline(0.0, color="#cbd5e1", linestyle="--", linewidth=1)
    ax_ret.set_title("Cumulative Return Over Steps", fontsize=10, fontweight="bold")
    ax_ret.set_xlabel("Step")
    ax_ret.set_ylabel("Cumulative Return")
    ax_ret.grid(alpha=0.25)
    ax_ret.legend(fontsize=8, loc="best")

    # --- Khung 3: So sánh điểm POI và đường hẹp ---
    if benchmark:
        names = [r["policy"].upper() for r in benchmark]
        scores = [float(r["mean_score"]) for r in benchmark]
        notes = [
            f"{r['mean_score']:.1f}\n(narrow {r.get('mean_narrow_m', 0):.0f}m, {r.get('success_rate', 0)*100:.0f}%)"
            for r in benchmark
        ]
        subtitle = "Benchmark Mean POI Score"
    else:
        names = [k.upper() for k in traces]
        scores = [float(v["summary"]["score"]) for v in traces.values()]
        notes = [
            f"{v['summary']['score']:.1f}\n(narrow {v['summary']['narrow_m']:.0f}m)"
            for v in traces.values()
        ]
        subtitle = "Episode POI Score"

    bar_colors = [palette.get(n.lower(), "#64748b") for n in names]
    bars = ax_bar.bar(names, scores, color=bar_colors, width=0.55)
    ax_bar.set_ylim(0, max(1.0, max(scores) * 1.25))
    for bar, note in zip(bars, notes):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(0.5, max(scores) * 0.02),
            note,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_bar.set_title(subtitle, fontsize=10, fontweight="bold")
    ax_bar.set_ylabel("POI Score")
    ax_bar.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
