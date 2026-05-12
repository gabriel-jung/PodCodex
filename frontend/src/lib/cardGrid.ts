/**
 * Card-grid template helpers.
 *
 * Per DESIGN.md §5 Grid: card grids use `repeat(auto-fill, minmax(<min>, 1fr))`
 * so wider viewports add columns instead of growing cards into billboards,
 * while leftover width between thresholds is absorbed by mild stretch (no
 * empty right-edge gap).
 */

/** Library tiles (HomePage). Slider 1 (big) to 5 (small). */
const SHOW_CARD_MIN_WIDTHS: Record<number, string> = {
  1: "1000px",
  2: "580px",
  3: "390px",
  4: "290px",
  5: "230px",
};

/** Episode tiles (ShowPage). Slider 2 (big) to 8 (small). */
const EPISODE_CARD_MIN_WIDTHS: Record<number, string> = {
  2: "560px",
  3: "380px",
  4: "290px",
  5: "230px",
  6: "190px",
  7: "165px",
  8: "145px",
};

export function showCardGridTemplate(size: number): string {
  const min = SHOW_CARD_MIN_WIDTHS[size] ?? SHOW_CARD_MIN_WIDTHS[3];
  return `repeat(auto-fill, minmax(${min}, 1fr))`;
}

export function episodeCardGridTemplate(size: number): string {
  const min = EPISODE_CARD_MIN_WIDTHS[size] ?? EPISODE_CARD_MIN_WIDTHS[4];
  return `repeat(auto-fill, minmax(${min}, 1fr))`;
}
