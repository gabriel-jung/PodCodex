/** Deterministic color assignment for speaker names.
 *  Six harmonious hues spaced around the color wheel, fixed lightness/chroma
 *  so they read the same in light and dark themes. Unknown speakers fall back
 *  to muted foreground. */

const SPEAKER_HUES = [25, 70, 155, 210, 280, 340];

function hashString(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619) >>> 0;
  return h;
}

const colorCache = new Map<string, string>();
const tintCache = new Map<string, string>();

/** CSS color for a speaker's accent. Stable across renders and reloads. */
export function speakerColor(name: string | null | undefined): string {
  if (!name) return "var(--muted-foreground)";
  const cached = colorCache.get(name);
  if (cached) return cached;
  const hue = SPEAKER_HUES[hashString(name) % SPEAKER_HUES.length];
  const color = `oklch(0.64 0.14 ${hue})`;
  colorCache.set(name, color);
  return color;
}

/** Faint background tint for a speaker chip. */
export function speakerTint(name: string | null | undefined): string {
  if (!name) return "transparent";
  const cached = tintCache.get(name);
  if (cached) return cached;
  const hue = SPEAKER_HUES[hashString(name) % SPEAKER_HUES.length];
  const tint = `oklch(0.64 0.14 ${hue} / 0.12)`;
  tintCache.set(name, tint);
  return tint;
}
