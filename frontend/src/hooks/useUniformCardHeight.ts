import { useLayoutEffect, type RefObject } from "react";

/**
 * Force every `[data-uniform-card]` descendant of `containerRef` to share the
 * tallest natural height. Keeps card layouts stable so artwork sits at the
 * same position regardless of per-card metadata variation.
 *
 * Uses a re-entrancy guard to avoid the classic ResizeObserver feedback loop:
 * clearing `minHeight` to remeasure shrinks cards back to natural, which would
 * fire RO again — we ignore RO callbacks while a measurement is in flight.
 */
export function useUniformCardHeight(
  containerRef: RefObject<HTMLElement | null>,
  deps: ReadonlyArray<unknown>,
) {
  useLayoutEffect(() => {
    const root = containerRef.current;
    if (!root) return;

    let measuring = false;

    const measure = () => {
      if (measuring) return;
      measuring = true;
      try {
        // Clear any min-heights we (or a previous render) set so cards that
        // no longer carry the marker (e.g. slider switched to horizontal
        // layout, button DOM reused) drop back to natural height.
        root.querySelectorAll<HTMLElement>("button").forEach((b) => {
          if (b.style.minHeight) b.style.minHeight = "";
        });
        const cards = Array.from(
          root.querySelectorAll<HTMLElement>("[data-uniform-card]"),
        );
        if (cards.length === 0) return;
        let max = 0;
        for (const c of cards) {
          const h = c.offsetHeight;
          if (h > max) max = h;
        }
        if (max > 0) {
          cards.forEach((c) => { c.style.minHeight = `${max}px`; });
        }
      } finally {
        requestAnimationFrame(() => { measuring = false; });
      }
    };

    measure();

    const ro = new ResizeObserver(() => measure());
    ro.observe(root);
    window.addEventListener("resize", measure);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", measure);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}
