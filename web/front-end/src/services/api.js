// Lớp gọi API. Đặt VITE_API_BASE để dùng back-end thật; để trống = chế độ MOCK (xem docs/API_CONTRACT.md).
import { mockPlan } from "./mockPlanner";
import { norm } from "../utils";

const BASE = import.meta.env.VITE_API_BASE || "";
export const MOCK = BASE === "";
const API = BASE === "/" ? "" : BASE;

let staticData = null;
export async function loadStatic() {
  if (staticData) return staticData;
  const get = async (f) => (await fetch("/data/" + f)).json();
  const [pois, hubs, ring] = await Promise.all([get("pois.json"), get("hubs.json"), get("central_ring.geojson")]);
  staticData = { pois, hubs, ring: ring.features[0].geometry.coordinates[0] };     // vòng ngoài [lng, lat]
  return staticData;
}

/** Gợi ý địa chỉ. Back-end thật: GET /api/geocode?q=... → [{label, lat, lng, id?}] */
export async function geocode(q) {
  if (!MOCK) {
    const r = await fetch(`${API}/api/geocode?q=${encodeURIComponent(q)}`);
    if (!r.ok) throw new Error("Geocode failed (" + r.status + ")");
    return r.json();
  }
  const { pois, hubs } = await loadStatic(), n = norm(q.trim());
  const pool = [...hubs.map((h) => ({ ...h, kind: "Hub" })), ...pois.map((p) => ({ ...p, kind: "POI" }))];
  return pool.filter((x) => !n || norm(x.name).includes(n)).slice(0, 8).map((x) => ({ id: x.id, label: x.name, lat: x.lat, lng: x.lng, kind: x.kind }));
}

/** Lập lộ trình. Back-end thật: POST /api/plan (xem docs/API_CONTRACT.md) */
export async function planTrip(req) {
  if (MOCK) return mockPlan({ start: req.start, end: req.end, departMin: req.departMin, budgetMin: req.budgetMin }, await loadStatic());
  const r = await fetch(`${API}/api/plan`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ start: req.start, end: req.end, departAt: req.departAt, budgetMin: req.budgetMin }),
  });
  if (!r.ok) throw new Error((await r.text()) || "Plan request failed (" + r.status + ")");
  return r.json();
}
