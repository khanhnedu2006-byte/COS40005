<script setup>
import { computed } from "vue";
import { state } from "../store";
import { BUDGET_PRESETS, slotOf, slotLabel, isBanSlot } from "../constants";
import { timeToMin, inPolygon, durFmt } from "../utils";
import PlaceInput from "./PlaceInput.vue";
defineEmits(["plan"]);
const slot = computed(() => slotOf(timeToMin(state.departAt)));
const warnings = computed(() => {
  const w = [];
  const ban = isBanSlot(slot.value);
  for (const [k, p] of [["Start", state.start], ["Destination", state.end]])
    if (p && ban && inPolygon(p.lat, p.lng, state.ring)) w.push({ level: "warn", text: `${k} is inside the central ring, where buses over 30 seats are banned during "${slotLabel(slot.value)}".` });
  if (ban) w.push({ level: "info", text: `Departure falls in a ring-ban window (${slotLabel(slot.value)}). Legs through the central ring are marked infeasible.` });
  return w;
});
const ok = computed(() => state.start && state.end && !state.loading);
function swap() { [state.start, state.end] = [state.end, state.start]; }
</script>
<template>
  <section>
    <h1>Plan a sightseeing trip</h1>
    <p class="note">Enter start (A), destination (B), departure time and a time budget. The AI chooses the sightseeing points the bus passes — the driver does not pick stops.</p>
    <PlaceInput v-model="state.start" label="Start (A)" placeholder="Search a hub or place in Da Nang" />
    <button class="btn btn-ghost swap" @click="swap">⇅ Swap A / B</button>
    <PlaceInput v-model="state.end" label="Destination (B)" placeholder="Search a hub or place" />
    <div class="row">
      <label>Departure<input type="time" v-model="state.departAt" /></label>
      <label>Budget (min)<input type="number" min="30" step="10" v-model.number="state.budgetMin" /></label>
    </div>
    <div class="presets"><button v-for="b in BUDGET_PRESETS" :key="b" :class="{ on: state.budgetMin === b }" @click="state.budgetMin = b">{{ durFmt(b) }}</button></div>
    <p class="note">Time slot: <strong>{{ slotLabel(slot) }}</strong></p>
    <p v-for="(w, i) in warnings" :key="i" :class="['msg', w.level]">{{ w.text }}</p>
    <p v-if="state.error" class="msg error">{{ state.error }}</p>
    <button class="btn btn-primary" :disabled="!ok" @click="$emit('plan')">{{ state.loading ? "Planning…" : "Plan route" }}</button>
  </section>
</template>
