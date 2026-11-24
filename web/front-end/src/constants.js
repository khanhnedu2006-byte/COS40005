// Khung giờ và giờ cấm – phải khớp src/busroute/utils/timeslots.py trong repo COS40005.
export const SLOTS = [
  { id: "morning", label: "Morning", from: 300, to: 630 },
  { id: "ban_noon", label: "Noon ban", from: 630, to: 720, ban: true },
  { id: "afternoon", label: "Afternoon", from: 720, to: 990 },
  { id: "ban_evening", label: "Evening ban", from: 990, to: 1140, ban: true },
  { id: "night", label: "Night", from: 1140, to: 1440 },
];
export const slotOf = (minuteOfDay) => {
  const m = ((Math.round(minuteOfDay) % 1440) + 1440) % 1440;
  const s = SLOTS.find((x) => m >= x.from && m < x.to);
  return s ? s.id : "night";          // trước 05:00 tính là night, giống timeslots.py
};
export const slotLabel = (id) => (SLOTS.find((s) => s.id === id) || { label: id }).label;
export const isBanSlot = (id) => id === "ban_noon" || id === "ban_evening";

export const ALGORITHMS = {
  ppo_gnn: { label: "AI route (PPO + GNN)", short: "PPO + GNN" },
  greedy: { label: "Greedy baseline", short: "Greedy" },
  shortest: { label: "Shortest path baseline", short: "Shortest" },
};
export const BUDGET_PRESETS = [90, 120, 180, 240, 360];
