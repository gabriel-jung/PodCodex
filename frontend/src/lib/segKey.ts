/**
 * Canonical segment identity used across the synthesis pipeline.
 *
 * The backend (`src/podcodex/core/_utils.py::seg_key`) mirrors this format
 * byte-for-byte. Timestamps are rounded to integer milliseconds to avoid
 * floating-point stringification drift (JS renders `1` as `"1"` while
 * Python renders `1.0` as `"1.0"` — any integer-second boundary would
 * otherwise silently fail the cross-process key lookup).
 */
export function segKey(seg: { speaker?: string; start: number; end: number }): string {
  return `${seg.speaker || ""}:${Math.round(seg.start * 1000)}:${Math.round(seg.end * 1000)}`;
}
