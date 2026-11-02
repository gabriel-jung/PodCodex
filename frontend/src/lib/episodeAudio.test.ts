import { describe, expect, it } from "vitest";
import type { Episode, ShowMeta } from "@/api/types";
import { canRedownloadAudio } from "./episodeAudio";

/**
 * Gates the delete-audio copy ("you can re-download it" vs "gone for good")
 * and every Download action. A false positive tells the user an original
 * recording can be fetched again, so the local cases matter most.
 */

const ep = (fields: Partial<Episode>) => ({ removed: false, audio_url: "", in_feed: true, ...fields }) as Episode;
const meta = (fields: Partial<ShowMeta>) => ({ rss_url: "", youtube_url: "", ...fields }) as ShowMeta;

describe("canRedownloadAudio", () => {
  it.each([
    ["an RSS item with an enclosure", ep({ audio_url: "https://x/a.mp3" }), meta({ rss_url: "https://x/feed" }), true],
    ["a YouTube video (no audio_url)", ep({}), meta({ youtube_url: "https://youtube.com/@x" }), true],
    ["a local episode", ep({ in_feed: false }), meta({}), false],
    ["a local file kept in an RSS show", ep({ in_feed: false }), meta({ rss_url: "https://x/feed" }), false],
    ["a local file with a stale audio_url in an RSS show", ep({ in_feed: false, audio_url: "https://x/a.mp3" }), meta({ rss_url: "https://x/feed" }), false],
    ["a local file dropped into a YouTube show", ep({ in_feed: false }), meta({ youtube_url: "https://youtube.com/@x" }), false],
    ["an item gone from the feed", ep({ audio_url: "https://x/a.mp3", removed: true }), meta({ rss_url: "https://x/feed" }), false],
    ["a removed YouTube video", ep({ removed: true }), meta({ youtube_url: "https://youtube.com/@x" }), false],
  ])("is right for %s", (_label, episode, show, expected) => {
    expect(canRedownloadAudio(episode, show)).toBe(expected);
  });

  it("is false while the show meta is still loading", () => {
    expect(canRedownloadAudio(ep({ audio_url: "https://x/a.mp3" }), undefined)).toBe(false);
  });
});
