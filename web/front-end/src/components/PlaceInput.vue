<script setup>
import { ref, watch } from "vue";
import { geocode } from "../services/api";
const props = defineProps({ modelValue: Object, label: String, placeholder: String });
const emit = defineEmits(["update:modelValue"]);
const q = ref(props.modelValue?.label || ""), items = ref([]), open = ref(false), err = ref("");
let timer = null;
watch(() => props.modelValue, (v) => { q.value = v?.label || ""; });
function onInput() {
  emit("update:modelValue", null); clearTimeout(timer);
  timer = setTimeout(search, 250);
}
async function search() {
  try { items.value = await geocode(q.value); err.value = ""; open.value = true; }
  catch (e) { items.value = []; err.value = e.message; }
}
function onBlur() { setTimeout(() => { open.value = false; }, 150); }
function pick(it) { emit("update:modelValue", it); q.value = it.label; open.value = false; }
</script>
<template>
  <div class="field">
    <span>{{ label }}</span>
    <input v-model="q" :placeholder="placeholder" autocomplete="off" @input="onInput" @focus="search" @blur="onBlur" />
    <ul v-if="open && items.length" class="sugg">
      <li v-for="it in items" :key="it.id || it.label" @mousedown.prevent="pick(it)">{{ it.label }}<small>{{ it.kind }}</small></li>
    </ul>
    <p v-if="err" class="msg error">{{ err }}</p>
  </div>
</template>
