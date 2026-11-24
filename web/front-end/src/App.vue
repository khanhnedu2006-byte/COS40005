<script setup>
import { onMounted, ref } from "vue";
import { state } from "./store";
import { loadStatic, planTrip, MOCK } from "./services/api";
import { timeToMin } from "./utils";
import TripSetup from "./components/TripSetup.vue";
import RouteResult from "./components/RouteResult.vue";
import MapView from "./components/MapView.vue";
const view = ref("setup");
onMounted(async () => { Object.assign(state, await loadStatic(), { ready: true }); });
async function plan() {
  state.loading = true; state.error = ""; state.progress = 0; state.playing = false;
  try {
    state.result = await planTrip({ start: state.start, end: state.end, departAt: state.departAt, departMin: timeToMin(state.departAt), budgetMin: state.budgetMin });
    state.algo = "ppo_gnn"; view.value = "result";
  } catch (e) { state.error = e.message || String(e); }
  finally { state.loading = false; }
}
</script>
<template>
  <div class="app">
    <div class="topbar"><div class="brand"><span class="brand-mark">◆</span>Da Nang Tour Bus<small>45-seat sightseeing route planner</small></div></div>
    <div v-if="MOCK" class="mockbar">Demo mode: routes come from a local mock planner using the real TomTom travel-time matrix. This is not the trained PPO + GNN model.</div>
    <div class="stage" v-if="state.ready">
      <div class="panel">
        <TripSetup v-if="view === 'setup'" @plan="plan" />
        <RouteResult v-else @back="view = 'setup'" />
      </div>
      <MapView />
    </div>
    <p v-else class="note" style="padding:20px">Loading…</p>
  </div>
</template>
