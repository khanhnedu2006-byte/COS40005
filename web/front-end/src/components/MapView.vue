<script setup>
import { onMounted, onBeforeUnmount, ref, watch, markRaw } from "vue";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { state, currentRoute } from "../store";
import { cumulative, pointAt } from "../utils";

const el = ref(null), tileFail = ref(false);
let map, layers, bus, ringLayer, poiLayer, raf, last = 0, pathCache = null;
const TILES = [
  { url: import.meta.env.VITE_TILE_URL || "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", att: import.meta.env.VITE_TILE_ATTRIBUTION || "Tiles © Esri" },
  { url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png", att: "© OpenStreetMap contributors" },
];
function addTiles(i = 0) {
  if (i >= TILES.length) { tileFail.value = true; return; }
  let errs = 0;
  const t = L.tileLayer(TILES[i].url, { attribution: TILES[i].att, maxZoom: 19 }).addTo(map);
  t.on("tileerror", () => { if (++errs === 6) { map.removeLayer(t); addTiles(i + 1); } });
}
const pin = (txt, cls = "") => L.divIcon({ className: "", html: `<div class="pin ${cls}">${txt}</div>`, iconSize: [26, 26], iconAnchor: [13, 13] });

onMounted(() => {
  map = markRaw(L.map(el.value).setView([16.06, 108.22], 11));
  addTiles();
  layers = L.layerGroup().addTo(map);
  ringLayer = L.polygon(state.ring.map(([lng, lat]) => [lat, lng]), { color: "#c0392b", weight: 2, dashArray: "6 6", fillOpacity: 0.06 }).addTo(map).bindTooltip("Central ring (buses > 30 seats banned in ban windows)");
  poiLayer = L.layerGroup().addTo(map);
  state.pois.forEach((p) => L.circleMarker([p.lat, p.lng], { radius: 5, color: "#0b6e75", weight: 1, fillOpacity: 0.6 }).bindTooltip(`${p.name} · ${p.score.toFixed(2)}`).addTo(poiLayer));
  state.hubs.forEach((h) => L.circleMarker([h.lat, h.lng], { radius: 6, color: "#1d2329", weight: 2, fillColor: "#fff", fillOpacity: 1 }).bindTooltip(h.name).addTo(poiLayer));
  setTimeout(() => map.invalidateSize(), 100);
  draw();
});
onBeforeUnmount(() => { cancelAnimationFrame(raf); map?.remove(); });

function legPath(route, i) {
  const l = route.legs[i];
  return l.geometry?.length ? l.geometry : [[route.nodes[i].lat, route.nodes[i].lng], [route.nodes[i + 1].lat, route.nodes[i + 1].lng]];
}
function draw() {
  layers.clearLayers(); bus = null; pathCache = null;
  const places = [state.start, state.end].filter(Boolean);
  const r = state.result && currentRoute();
  if (!r || !r.feasible) {
    places.forEach((p, i) => L.marker([p.lat, p.lng], { icon: pin(i ? "B" : "A", i ? "se" : "") }).addTo(layers));
    if (places.length) map.fitBounds(L.latLngBounds(places.map((p) => [p.lat, p.lng])).pad(0.4));
    return;
  }
  const full = [];
  r.legs.forEach((l, i) => {
    const pts = legPath(r, i); full.push(...pts);
    L.polyline(pts, { color: l.throughRing ? "#b26a00" : "#0b6e75", weight: 5, opacity: 0.9, dashArray: l.geometry ? null : "8 8" }).addTo(layers);
  });
  r.nodes.forEach((n, i) => L.marker([n.lat, n.lng], { icon: pin(n.type === "start" ? "A" : n.type === "end" ? "B" : i, n.type === "end" ? "se" : "") }).bindTooltip(n.name).addTo(layers));
  pathCache = { pts: full, cum: cumulative(full) };
  map.fitBounds(L.latLngBounds(full).pad(0.15));
  moveBus();
}
function moveBus() {
  if (!pathCache) return;
  const p = pointAt(pathCache.pts, pathCache.cum, state.progress);
  if (!bus) bus = L.marker(p, { icon: L.divIcon({ className: "", html: '<div class="bus-ic">🚌</div>', iconSize: [28, 28], iconAnchor: [14, 14] }), zIndexOffset: 1000 }).addTo(layers);
  else bus.setLatLng(p);
}
function tick(ts) {
  if (!state.playing) return;
  if (last) state.progress = Math.min(1, state.progress + (ts - last) / 20000);
  last = ts; if (state.progress >= 1) state.playing = false; else raf = requestAnimationFrame(tick);
}
watch(() => state.playing, (v) => { last = 0; if (v) { if (state.progress >= 1) state.progress = 0; raf = requestAnimationFrame(tick); } });
watch(() => state.progress, moveBus);
watch(() => [state.result, state.algo, state.start, state.end], draw);
</script>
<template>
  <div class="mapwrap"><div ref="el" class="map"></div><div v-if="tileFail" class="tilewarn">Map tiles could not load — check your network.</div></div>
</template>
