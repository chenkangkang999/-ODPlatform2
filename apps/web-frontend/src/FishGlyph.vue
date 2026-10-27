<script setup>
import { computed } from 'vue'

const props = defineProps({
  variant: { type: Number, default: 0 },
  size: { type: Number, default: 36 },
})

// Decorative silhouettes only: the variants do not identify particular species.
const silhouettes = [
  {
    body: 'M12 20C20 9 36 8 47 19Q49 20 47 22C36 32 21 31 12 20Z',
    tail: 'M15 20 4 11 6 20 4 29Z',
    fins: ['M23 12 29 5 34 11', 'M26 28 32 34 36 27'],
    lines: ['M36 14Q31 20 36 26', 'M25 18 19 21 26 24'],
    eye: [42, 18],
  },
  {
    body: 'M15 20C15 10 25 5 35 9S49 18 46 24 36 33 27 31 15 26 15 20Z',
    tail: 'M17 20 5 13 6 27Z',
    fins: ['M23 10 26 3 34 9', 'M23 29 29 36 35 30'],
    lines: ['M36 12Q31 19 36 28', 'M25 16Q20 21 28 24'],
    eye: [41, 18],
  },
  {
    body: 'M12 21C23 14 39 13 49 19Q52 21 48 23C36 28 23 27 12 21Z',
    tail: 'M15 21 3 15 7 21 3 27Z',
    fins: ['M27 16 32 10 39 15', 'M27 25 34 31 39 25'],
    lines: ['M40 17Q37 21 40 25', 'M24 20 19 22 26 23'],
    eye: [45, 19],
  },
  {
    body: 'M12 21C16 12 31 10 44 15Q50 17 50 21C49 28 31 29 19 25Z',
    tail: 'M16 21 4 12 7 21 4 30Z',
    fins: ['M24 14 26 7 33 12', 'M27 27 33 33 39 27'],
    lines: ['M39 15Q35 21 40 27', 'M46 24Q50 27 54 25', 'M45 25Q48 31 52 30', 'M24 19 19 24 28 23'],
    eye: [44, 18],
  },
  {
    body: 'M14 20C22 9 29 6 36 10Q44 14 47 20C42 27 36 33 29 31S18 25 14 20Z',
    tail: 'M17 20 5 10 7 20 5 30Z',
    fins: ['M24 11 29 3 33 9', 'M23 28 28 37 33 31'],
    lines: ['M37 12Q33 20 38 28', 'M27 15 24 24 31 22'],
    eye: [41, 18],
  },
  {
    body: 'M10 22C18 13 36 12 48 18L52 21 47 24C34 28 18 27 10 22Z',
    tail: 'M14 22 3 16 7 22 3 29Z',
    fins: ['M32 15 38 7 41 16', 'M29 26 35 32 40 25'],
    lines: ['M42 17Q39 21 43 25', 'M30 21 23 24 33 24'],
    eye: [47, 19],
  },
  {
    body: 'M14 19C20 11 34 9 44 14Q51 18 47 23C37 31 22 28 14 19Z',
    tail: 'M18 20 5 10 9 20 3 29 16 25Z',
    fins: ['M22 13 24 7 30 11 34 6 38 12', 'M26 27 32 34 36 28'],
    lines: ['M38 13Q33 19 38 26', 'M24 17 21 22 28 23', 'M29 13 27 16', 'M33 12 31 15'],
    eye: [43, 17],
  },
  {
    body: 'M14 21C18 13 28 11 38 14Q46 16 49 21C44 29 24 30 14 21Z',
    tail: 'M17 21 4 16 7 22 4 29 17 25Z',
    fins: ['M26 14 25 5 35 13', 'M29 28 36 35 39 26'],
    lines: ['M39 15Q35 21 39 27', 'M25 18 20 23 28 24', 'M27 12Q35 10 40 14'],
    eye: [43, 19],
  },
  {
    body: 'M13 20C21 11 33 10 44 15Q49 17 50 21L45 24C34 31 20 29 13 20Z',
    tail: 'M17 20 4 9 8 20 4 32 17 25Z',
    fins: ['M23 13 22 6 31 10 36 5 39 13', 'M25 28 29 35 35 29'],
    lines: ['M38 14Q34 21 39 27', 'M22 18 19 22 27 23', 'M29 14Q27 20 29 26'],
    eye: [44, 18],
  },
  {
    body: 'M13 21C19 15 30 11 42 14Q48 16 50 20C46 26 38 29 28 28S17 24 13 21Z',
    tail: 'M17 21 4 13 5 21 3 30 16 25Z',
    fins: ['M29 14 33 6 39 14', 'M26 27 32 34 38 27'],
    lines: ['M39 15Q35 21 39 26', 'M25 20 20 24 29 24', 'M19 19Q26 15 33 16'],
    eye: [44, 18],
  },
]

const index = computed(() => ((Math.trunc(props.variant) || 0) % silhouettes.length + silhouettes.length) % silhouettes.length)
const fish = computed(() => silhouettes[index.value])
const palette = computed(() => [
  { stroke: '#438cab', fill: '#c6e7f3', fin: '#e0f2fa' },
  { stroke: '#4d95a8', fill: '#c6e8ec', fin: '#e0f4f3' },
  { stroke: '#5489b6', fill: '#d2e7f8', fin: '#e7f3fc' },
][index.value % 3])
</script>

<template>
  <svg
    class="fish-glyph"
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 56 40"
    :width="size"
    :height="size"
    aria-hidden="true"
    focusable="false"
  >
    <g :stroke="palette.stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
      <path v-for="(fin, i) in fish.fins" :key="`fin-${i}`" :d="fin" :fill="palette.fin" />
      <path :d="fish.tail" :fill="palette.fin" />
      <path :d="fish.body" :fill="palette.fill" />
      <path v-for="(line, i) in fish.lines" :key="`line-${i}`" :d="line" fill="none" :stroke-opacity="i > 1 ? .65 : .85" />
    </g>
    <circle :cx="fish.eye[0]" :cy="fish.eye[1]" r="1.25" :fill="palette.stroke" />
    <circle :cx="fish.eye[0] + .35" :cy="fish.eye[1] - .4" r=".35" fill="#fff" />
  </svg>
</template>
