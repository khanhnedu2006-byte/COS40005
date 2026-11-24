<script setup>
import { computed } from "vue";
import { state, currentRoute } from "../store";
import { ALGORITHMS, slotLabel } from "../constants";
import { clockFmt, durFmt, timeToMin, km } from "../utils";
defineEmits(["back"]);
const route = computed(currentRoute);
const depart = computed(() => timeToMin(state.departAt));
const names = computed(() => Object.fromEntries(state.pois.concat(state.hubs).map((n) => [n.id, n.name])));
const dist = computed(() => (route.value?.legs || []).reduce((s, l) => s + (l.distanceM || 0), 0));
</script>
<template>
  <section v-if="state.result">
    <button class="btn btn-ghost swap" @click="$emit('back')">← Edit trip</button>
    <p v-for="(i, k) in state.result.validation?.issues || []" :key="k" :class="['msg', i.level]">{{ i.text }}</p>
    <p class="note">Slot: <strong>{{ slotLabel(state.result.slot) }}</strong><span v-if="state.result.directMin"> · direct A→B {{ durFmt(state.result.directMin) }}</span></p>
    <div class="tabs"><button v-for="r in state.result.routes" :key="r.algorithm" :class="{ on: state.algo === r.algorithm }" @click="state.algo = r.algorithm; state.progress = 0">{{ ALGORITHMS[r.algorithm]?.short || r.algorithm }}</button></div>
    <template v-if="route">
      <p v-if="!route.feasible" class="msg error">{{ route.reason || "No feasible route." }}</p>
      <template v-else>
        <div class="metrics">
          <div class="metric"><b>{{ route.totalScore.toFixed(2) }}</b><span>Total POI score</span></div>
          <div class="metric"><b>{{ route.poiCount }}</b><span>POIs passed</span></div>
          <div class="metric"><b>{{ durFmt(route.totalMin) }}</b><span>of {{ durFmt(route.budgetMin) }}</span></div>
        </div>
        <div class="bar"><div class="bar-fill" :style="{ width: Math.min(100, (route.totalMin / route.budgetMin) * 100) + '%' }"></div></div>
        <p class="budget-sub">{{ km(dist) }} · arrive {{ clockFmt(depart + route.totalMin) }}</p>
        <div class="play">
          <button class="btn btn-primary" @click="state.playing = !state.playing">{{ state.playing ? "Pause" : "Preview" }}</button>
          <input type="range" min="0" max="1" step="0.001" v-model.number="state.progress" style="flex:1" />
        </div>
        <ol class="tl">
          <template v-for="(n, i) in route.nodes" :key="n.id + i">
            <li>
              <time>{{ clockFmt(depart + n.atMin) }}</time>
              <div><div class="nm">{{ n.type === "start" ? "A · " : n.type === "end" ? "B · " : "" }}{{ n.name }}</div>
                <div class="sub"><span v-if="n.type === 'poi'">score {{ n.score.toFixed(2) }}</span></div></div>
            </li>
            <li v-if="route.legs[i]" style="background:#fafbfb">
              <time style="font-weight:400;color:var(--muted)">{{ durFmt(route.legs[i].durationMin) }}</time>
              <div class="sub">
                <span v-if="route.legs[i].throughRing" class="tag ring">central ring</span>
                <span v-if="route.legs[i].poisPassed?.length">also passes: {{ route.legs[i].poisPassed.map((q) => names[q] || q).join(", ") }}</span>
                <span v-else>no extra POIs on this leg</span>
              </div>
            </li>
          </template>
        </ol>
      </template>
    </template>
    <h2>Comparison</h2>
    <table class="cmp"><thead><tr><th>Method</th><th>Score</th><th>POIs</th><th>Time</th></tr></thead><tbody>
      <tr v-for="r in state.result.routes" :key="r.algorithm"><td>{{ ALGORITHMS[r.algorithm]?.short }}</td><td>{{ r.feasible ? r.totalScore.toFixed(2) : "–" }}</td><td>{{ r.feasible ? r.poiCount : "–" }}</td><td>{{ r.feasible ? durFmt(r.totalMin) : "–" }}</td></tr>
    </tbody></table>
  </section>
</template>
