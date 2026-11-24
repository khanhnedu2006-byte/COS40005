// MOCK planner chạy trong trình duyệt, dùng ma trận thật data/od_matrix.csv (55 điểm × 5 khung giờ).
// Chỉ để demo giao diện khi chưa có back-end. KHÔNG phải model PPO + GNN:
//   - "ppo_gnn"  = tạm dùng greedy ngẫu nhiên nhiều lần, lấy lộ trình tốt nhất
//   - "greedy"   = chọn điểm có (điểm POI / phút) lớn nhất
//   - "shortest" = đi thẳng A → B
// Mock chỉ hỗ trợ A, B là một trong 55 điểm đã biết (POI hoặc hub).
import { slotOf } from "../constants";

let matrix = null;      // key "slot|from|to" → {dur (phút), dist (m), feas, passed[]}

export async function loadMatrix() {
  if (matrix) return matrix;
  const text = (await (await fetch("/data/od_matrix.csv")).text()).replace(/^﻿/, "");
  const lines = text.trim().split(/\r?\n/), head = lines[0].split(","), col = (n) => head.indexOf(n);
  const iF = col("from_id"), iT = col("to_id"), iS = col("slot"), iD = col("duration_s"), iM = col("distance_m"), iOK = col("feasible"), iP = col("pois_passed");
  matrix = new Map();
  for (let k = 1; k < lines.length; k++) {
    const c = lines[k].split(",");
    matrix.set(`${c[iS]}|${c[iF]}|${c[iT]}`, { dur: +c[iD] / 60, dist: +c[iM], feas: c[iOK] === "1", passed: c[iP] ? c[iP].split(";") : [] });
  }
  return matrix;
}

function rng(seed) { return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

export async function mockPlan({ start, end, departMin, budgetMin }, ctx) {
  const M = await loadMatrix();
  const byId = new Map([...ctx.pois, ...ctx.hubs].map((n) => [n.id, n]));
  const score = new Map(ctx.pois.map((p) => [p.id, p.score]));
  const leg = (t, a, b) => M.get(`${slotOf(departMin + t)}|${a}|${b}`);
  const A = start.id, B = end.id;

  function run(noise, rand) {
    let cur = A, t = 0; const seen = new Set(score.has(A) ? [A] : []);
    const seq = [{ id: A, at: 0 }], legs = [];
    for (;;) {
      let best = null;
      for (const p of ctx.pois) {
        if (p.id === cur || p.id === B || seen.has(p.id)) continue;
        const l = leg(t, cur, p.id); if (!l || !l.feas) continue;
        const back = leg(t + l.dur, p.id, B); if (!back || !back.feas) continue;     // action mask: phải còn kịp về B
        if (t + l.dur + back.dur > budgetMin) continue;
        const gain = score.get(p.id) + l.passed.filter((q) => !seen.has(q) && q !== p.id).reduce((s, q) => s + (score.get(q) || 0), 0);
        const v = (gain / Math.max(l.dur, 1)) * (1 + noise * (rand() - 0.5));
        if (!best || v > best.v) best = { v, p, l };
      }
      if (!best) break;
      best.l.passed.forEach((q) => score.has(q) && seen.add(q)); seen.add(best.p.id);
      t += best.l.dur; cur = best.p.id; seq.push({ id: cur, at: t });
      legs.push({ ...best.l, slot: slotOf(departMin + t - best.l.dur) });
    }
    const fin = leg(t, cur, B);
    if (!fin || !fin.feas || t + fin.dur > budgetMin) return null;
    fin.passed.forEach((q) => score.has(q) && seen.add(q));
    legs.push({ ...fin, slot: slotOf(departMin + t) }); t += fin.dur; seq.push({ id: B, at: t });
    return { seq, legs, total: t };
  }

  function build(algorithm, label, r) {
    if (!r) return { algorithm, label, feasible: false, totalMin: 0, budgetMin, totalScore: 0, poiCount: 0, nodes: [], legs: [], reason: "No feasible route within the budget" };
    const nodes = r.seq.map((s, i) => { const n = byId.get(s.id); return { id: n.id, type: i === 0 ? "start" : i === r.seq.length - 1 ? "end" : "poi", name: n.name, lat: n.lat, lng: n.lng, score: score.get(n.id) || 0, atMin: s.at }; });
    const shown = new Set(); if (score.has(nodes[0].id)) shown.add(nodes[0].id);
    const legs = r.legs.map((l, i) => {
      const dest = nodes[i + 1].id, fresh = l.passed.filter((q) => q !== dest && !shown.has(q));
      fresh.forEach((q) => shown.add(q)); shown.add(dest);
      return { fromIndex: i, durationMin: l.dur, distanceM: l.dist, throughRing: false, slot: l.slot, poisPassed: fresh, geometry: null };
    });
    const counted = new Set(); nodes.forEach((n, i) => { if (i > 0 && score.has(n.id)) counted.add(n.id); }); legs.forEach((l) => l.poisPassed.forEach((q) => counted.add(q)));
    const totalScore = [...counted].reduce((s, q) => s + (score.get(q) || 0), 0);
    return { algorithm, label, feasible: true, totalMin: r.total, budgetMin, totalScore, poiCount: counted.size, nodes, legs };
  }

  const direct = leg(0, A, B);
  const shortest = direct && direct.feas ? { seq: [{ id: A, at: 0 }, { id: B, at: direct.dur }], legs: [{ ...direct, slot: slotOf(departMin) }], total: direct.dur } : null;
  const greedy = run(0, Math.random);
  let ai = greedy, aiScore = -1; const rand = rng(2026);
  const scoreOf = (r) => (r ? build("x", "x", r).totalScore : -1);
  aiScore = scoreOf(ai);
  for (let k = 0; k < 150; k++) { const r = run(0.6, rand); const s = scoreOf(r); if (s > aiScore) { ai = r; aiScore = s; } }

  const issues = [];
  if (!direct || !direct.feas) issues.push({ level: "error", text: "A → B is not reachable in this time slot (central ring ban)." });
  else if (direct.dur > budgetMin) issues.push({ level: "error", text: `Budget is shorter than the direct trip (${Math.ceil(direct.dur)} min).` });
  return {
    mock: true, slot: slotOf(departMin), directMin: direct && direct.feas ? direct.dur : null, validation: { issues },
    routes: [build("ppo_gnn", "AI route (PPO + GNN)", ai), build("greedy", "Greedy baseline", greedy), build("shortest", "Shortest path baseline", shortest)],
  };
}
