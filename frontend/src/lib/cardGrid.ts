/**
 * Card-grid template helpers.
 *
 * Per DESIGN.md §5 Grid: card grids use `repeat(auto-fill, minmax(<min>, 1fr))`
 * so wider viewports add columns instead of growing cards into billboards,
 * while leftover width between thresholds is absorbed by mild stretch (no
 * empty right-edge gap).
 *
 * Minimums are stored as numbers so a virtualized grid can reproduce the
 * column count `auto-fill` would have picked. Keep the two in sync by always
 * going through these helpers.
 */

/** Library tiles (HomePage). Slider 1 (big) to 5 (small). */
const SHOW_CARD_MIN_WIDTHS: Record<number, number> = {
  1: 1000,
  2: 580,
  3: 390,
  4: 290,
  5: 230,
};

/** Episode tiles (ShowPage). Slider 2 (big) to 8 (small). */
const EPISODE_CARD_MIN_WIDTHS: Record<number, number> = {
  2: 560,
  3: 380,
  4: 290,
  5: 230,
  6: 190,
  7: 165,
  8: 145,
};

/** Minimum episode-tile width in px for a slider position. */
export function episodeCardMinWidth(size: number): number {
  return EPISODE_CARD_MIN_WIDTHS[size] ?? EPISODE_CARD_MIN_WIDTHS[4];
}

/**
 * Columns `repeat(auto-fill, minmax(min, 1fr))` yields at `width` px of
 * content box. Mirrors the CSS so a virtualized grid lays out identically.
 */
export function autoFillColumns(width: number, min: number, gap: number): number {
  if (width <= 0) return 1;
  return Math.max(1, Math.floor((width + gap) / (min + gap)));
}

export function showCardGridTemplate(size: number): string {
  const min = SHOW_CARD_MIN_WIDTHS[size] ?? SHOW_CARD_MIN_WIDTHS[3];
  return `repeat(auto-fill, minmax(${min}px, 1fr))`;
}
