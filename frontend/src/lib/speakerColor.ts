/** Deterministic color assignment for speaker names.
 *  Eleven hues spread around the wheel; fixed chroma/lightness so they read
 *  the same in light and dark themes. Unknown speakers fall back to muted
 *  foreground. The hash runs FNV-1a + a splitmix finalizer — the finalizer
 *  is load-bearing for short names (David/Olivier/Alice hashed all to the
 *  same bucket with raw FNV). */

const SPEAKER_HUES = [10, 40, 70, 130, 165, 200, 235, 265, 295, 325, 355];
const SPEAKER_L = 0.62;
const SPEAKER_C = 0.17;

function hashString(s: string): number {
  let h = 2166136261 >>> 0;
  for (const c of s) {
    h = Math.imul(h ^ (c.codePointAt(0) ?? 0), 16777619) >>> 0;
  }
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b) >>> 0;
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35) >>> 0;
  h ^= h >>> 16;
  return h >>> 0;
}

const colorCache = new Map<string, string>();
const tintCache = new Map<string, string>();

function hueFor(name: string): number {
  return SPEAKER_HUES[hashString(name) % SPEAKER_HUES.length];
}

/** CSS color for a speaker's accent. Stable across renders and reloads. */
export function speakerColor(name: string | null | undefined): string {
  if (!name) return "var(--muted-foreground)";
  const cached = colorCache.get(name);
  if (cached) return cached;
  const color = `oklch(${SPEAKER_L} ${SPEAKER_C} ${hueFor(name)})`;
  colorCache.set(name, color);
  return color;
}

/** Faint background tint for a speaker chip. */
export function speakerTint(name: string | null | undefined): string {
  if (!name) return "transparent";
  const cached = tintCache.get(name);
  if (cached) return cached;
  const tint = `oklch(${SPEAKER_L} ${SPEAKER_C} ${hueFor(name)} / 0.12)`;
  tintCache.set(name, tint);
  return tint;
}
