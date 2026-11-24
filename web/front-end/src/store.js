import { reactive } from "vue";
export const state = reactive({
  ready: false, pois: [], hubs: [], ring: [],
  start: null, end: null, departAt: "08:00", budgetMin: 180,
  loading: false, error: "", result: null, algo: "ppo_gnn", progress: 0, playing: false,
});
export const currentRoute = () => state.result?.routes.find((r) => r.algorithm === state.algo) || null;
