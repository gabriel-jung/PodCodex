import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { formatBytes, formatDate, formatDuration, formatTime, languageToISO, stripHtml } from "./utils";

describe("formatTime", () => {
  it("carries rounded hundredths into the next second", () => {
    expect(formatTime(5.996)).toBe("00:06.00");
    expect(formatTime(5.994)).toBe("00:05.99");
  });

  it("shows hours only when there are some, and truncates without hundredths", () => {
    expect(formatTime(3725.5, false)).toBe("1:02:05");
    expect(formatTime(59.9, false)).toBe("00:59");
  });

  it("reads a missing time as zero", () => {
    expect(formatTime(null)).toBe("00:00");
  });
});

describe("formatDate", () => {
  const savedTz = process.env.TZ;
  beforeEach(() => {
    process.env.TZ = "America/Los_Angeles";
  });
  afterEach(() => {
    process.env.TZ = savedTz;
  });
  const day = (s: string) => new Date(s).getUTCDate();

  it("keeps a calendar date on its day west of UTC", () => {
    for (const s of ["2026-10-01", "2026-10-01T00:00:00Z", "2026-10-01T00:00:00+00:00"]) {
      expect(formatDate(s)).toContain(String(day("2026-10-01")));
      expect(formatDate(s)).not.toContain("30");
    }
  });

  it("reads a full timestamp in local time", () => {
    // 03:00 UTC on Oct 1 is still Sep 30 in Los Angeles.
    expect(formatDate("2026-10-01T03:00:00Z")).toContain("30");
  });

  it("keeps an offset-less local midnight on its day east of UTC", () => {
    process.env.TZ = "Europe/Paris";
    expect(formatDate("2026-10-01T00:00:00")).not.toContain("30");
  });

  it("passes an unparseable value through", () => {
    expect(formatDate("soon")).toBe("soon");
  });
});

describe("formatDuration and formatBytes", () => {
  it("formats durations and sizes", () => {
    expect(formatDuration(3720)).toBe("1h 2m");
    expect(formatDuration(0)).toBe("");
    expect(formatBytes(12345678)).toBe("11.8 MB");
  });
});

describe("languageToISO", () => {
  it("maps language names, Polish included", () => {
    expect(languageToISO("Polish")).toBe("pl");
    expect(languageToISO(" French ")).toBe("fr");
    expect(languageToISO("xx")).toBe("xx");
  });
});

describe("stripHtml", () => {
  it("turns an invalid code point into U+FFFD instead of throwing", () => {
    expect(stripHtml("a &#99999999; b &#x110000;")).toBe("a \uFFFD b \uFFFD");
  });

  it("drops tags and decodes named and numeric entities once", () => {
    expect(stripHtml("<p>It&#8217;s <b>a</b> &amp;lt;test&#x21;</p>")).toBe("It\u2019s a &lt;test!");
  });
});
