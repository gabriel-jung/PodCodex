import { describe, expect, it } from "vitest";
import { isEdited, plainStatus, reviewStatus, translationsStatus } from "./stepStatus";

/**
 * The frontend half of "edited beats freshness". These three derive every
 * StageCard and panel header status, so a change to `isEdited` or to the
 * translations reduction silently relabels every episode in the app.
 * The backend half is pinned by tests/test_versions.py.
 */

describe("isEdited", () => {
  it.each([
    ["a hand edit", { manual_edit: true }, true],
    ["a validated version", { type: "validated" }, true],
    ["a raw auto version", { manual_edit: false, type: "raw" }, false],
    ["an empty provenance", {}, false],
    ["no provenance at all", null, false],
    ["undefined", undefined, false],
  ])("reads %s as %s", (_label, provenance, expected) => {
    expect(isEdited(provenance)).toBe(expected);
  });
});

describe("reviewStatus", () => {
  it("is none when the step has no content, whatever the provenance says", () => {
    expect(reviewStatus(false, { manual_edit: true })).toBe("none");
  });

  it("is ready for an edited version and review for a raw one", () => {
    expect(reviewStatus(true, { manual_edit: true })).toBe("ready");
    expect(reviewStatus(true, { type: "validated" })).toBe("ready");
    expect(reviewStatus(true, { type: "raw" })).toBe("review");
    expect(reviewStatus(true, null)).toBe("review");
  });
});

describe("plainStatus", () => {
  it("has no review state: content or nothing", () => {
    expect(plainStatus(true)).toBe("ready");
    expect(plainStatus(false)).toBe("none");
  });
});

describe("translationsStatus", () => {
  it("is none with no translations", () => {
    expect(translationsStatus([], {})).toBe("none");
    expect(translationsStatus([], null)).toBe("none");
  });

  it("is ready only when every language is edited", () => {
    expect(translationsStatus(["fr"], { fr: { manual_edit: true } })).toBe("ready");
    expect(
      translationsStatus(["fr", "de"], {
        fr: { manual_edit: true },
        de: { type: "validated" },
      }),
    ).toBe("ready");
  });

  it("is review when any one language still needs it", () => {
    expect(
      translationsStatus(["fr", "de"], {
        fr: { manual_edit: true },
        de: { type: "raw" },
      }),
    ).toBe("review");
  });

  it("treats a language with no provenance row as needing review", () => {
    expect(translationsStatus(["fr"], {})).toBe("review");
    expect(translationsStatus(["fr"], null)).toBe("review");
  });
});
