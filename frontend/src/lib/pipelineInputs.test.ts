import { describe, expect, it } from "vitest";
import type { VersionEntry } from "@/api/types";
import { sortVersionsForDefault } from "./pipelineInputs";

/** The picker's default must be the version the backend runs on when nothing
 *  is pinned (`core/versions.py` `_source_candidates`), or the run uses
 *  different text from the one shown. Inputs are newest first, as the API
 *  returns them. */

const v = (id: string, step: string, edited = false) =>
  ({ id, step, manual_edit: edited }) as unknown as VersionEntry;

describe("sortVersionsForDefault", () => {
  it("lets a fresh transcribe supersede an older edited transcript", () => {
    const fresh = v("fresh", "transcript");
    const edited = v("edited", "transcript", true);
    expect(sortVersionsForDefault([fresh, edited], "correct")[0].id).toBe("fresh");
  });

  it("puts the verified version first", () => {
    const fresh = v("fresh", "transcript");
    const edited = v("edited", "transcript", true);
    const sorted = sortVersionsForDefault([fresh, edited], "correct", { step: "transcript", version_id: "edited" });
    expect(sorted[0].id).toBe("edited");
  });

  it("keeps corrected versions edited first, above any transcript", () => {
    const sorted = sortVersionsForDefault(
      [v("t", "transcript"), v("c-raw", "corrected"), v("c-edited", "corrected", true)],
      "index",
    );
    expect(sorted.map((x) => x.id)).toEqual(["c-edited", "c-raw", "t"]);
  });
});
