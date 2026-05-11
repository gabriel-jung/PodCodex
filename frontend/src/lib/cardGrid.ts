/**
 * Card-grid template helpers.
 *
 * Per DESIGN.md §5 Grid: card grids use `repeat(auto-fill, <px>)` so wider
 * viewports add columns instead of stretching the cards the user picked.
 */

/** Library tiles (HomePage). Slider 1 (big) → 5 (small). */
const SHOW_CARD_WIDTHS: Record<number, string> = {
  1: "1000px",
  2: "580px",
  3: "390px",
  4: "290px",
  5: "230px",
};

/** Episode tiles (ShowPage). Slider 2 (big) → 8 (small). */
const EPISODE_CARD_WIDTHS: Record<number, string> = {
  2: "560px",
  3: "380px",
  4: "290px",
  5: "230px",
  6: "190px",
  7: "165px",
  8: "145px",
};

export function showCardGridTemplate(size: number): string {
  return `repeat(auto-fill, ${SHOW_CARD_WIDTHS[size] ?? SHOW_CARD_WIDTHS[3]})`;
}

export function episodeCardGridTemplate(size: number): string {
  return `repeat(auto-fill, ${EPISODE_CARD_WIDTHS[size] ?? EPISODE_CARD_WIDTHS[4]})`;
}
