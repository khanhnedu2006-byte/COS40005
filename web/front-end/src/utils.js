export const pad = (n) => String(n).padStart(2, "0");
export const clockFmt = (m) => { m = ((Math.round(m) % 1440) + 1440) % 1440; return pad(Math.floor(m / 60)) + ":" + pad(m % 60); };
export const durFmt = (m) => { m = Math.max(0, Math.round(m)); return (m >= 60 ? Math.floor(m / 60) + "h " : "") + (m % 60) + "m"; };
export const timeToMin = (t) => { const [h, m] = String(t || "08:00").split(":").map(Number); return h * 60 + m; };
export const km = (m) => (m / 1000).toFixed(1) + " km";
export const norm = (s) => String(s).normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/đ/g, "d").replace(/Đ/g, "D").toLowerCase();

export function haversine(a, b) {            // mét, điểm dạng [lat, lng]
  const R = 6371000, rad = Math.PI / 180, dLat = (b[0] - a[0]) * rad, dLng = (b[1] - a[1]) * rad;
  const s = Math.sin(dLat / 2) ** 2 + Math.cos(a[0] * rad) * Math.cos(b[0] * rad) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
}
export function cumulative(coords) { const c = [0]; for (let i = 1; i < coords.length; i++) c.push(c[i - 1] + haversine(coords[i - 1], coords[i])); return c; }
export function pointAt(coords, cum, frac) {
  const total = cum[cum.length - 1] || 1, len = Math.max(0, Math.min(1, frac)) * total;
  for (let i = 1; i < coords.length; i++) if (len <= cum[i]) {
    const t = (len - cum[i - 1]) / ((cum[i] - cum[i - 1]) || 1);
    return [coords[i - 1][0] + t * (coords[i][0] - coords[i - 1][0]), coords[i - 1][1] + t * (coords[i][1] - coords[i - 1][1])];
  }
  return coords[coords.length - 1];
}
// Ray casting; ring = mảng [lng, lat] (GeoJSON)
export function inPolygon(lat, lng, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if ((yi > lat) !== (yj > lat) && lng < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
