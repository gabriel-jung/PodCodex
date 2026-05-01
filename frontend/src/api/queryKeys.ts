/**
 * Central TanStack Query key factory.
 *
 * One source of truth for every cache key used by useQuery / invalidateQueries.
 * Avoids string typos, documents the cache namespace, and lets us refactor
 * key shapes in one place.
 *
 * Key shapes mirror the strings they replace — do not reorder tuple elements
 * without checking callers, since React Query matches by prefix.
 */

type AudioPath = string | null | undefined;

export const queryKeys = {
  // ── System / config ────────────────────────────────────
  health: () => ["health"] as const,
  config: () => ["config"] as const,
  pipelineConfig: () => ["pipeline-config"] as const,
  secrets: () => ["config", "secrets"] as const,
  capabilities: () => ["system", "extras"] as const,
  models: () => ["models"] as const,
  gpuStatus: () => ["gpu", "status"] as const,

  // ── Shows & episodes ───────────────────────────────────
  shows: () => ["shows"] as const,
  showMeta: (folder: string) => ["showMeta", folder] as const,
  speakerRoster: (folder: string) => ["speakerRoster", folder] as const,
  /** All episodes — broad invalidation (every folder, every defaults). */
  episodesAll: () => ["episodes"] as const,
  /** Episodes for a specific show folder. */
  episodesForFolder: (folder: string) => ["episodes", folder] as const,
  /** Episodes for a folder + specific pipeline defaults (the fully-specified key). */
  episodes: (folder: string, pipelineDefaults: unknown) =>
    ["episodes", folder, pipelineDefaults] as const,

  // ── Per-episode pipeline data ──────────────────────────
  transcribeSegments: (audioPath: AudioPath) =>
    ["transcribe", "segments", audioPath] as const,
  speakerMap: (audioPath: AudioPath) => ["speaker-map", audioPath] as const,
  /** All versions across all steps for an episode (unified endpoint). */
  allVersions: (audioPath: AudioPath) => ["versions", "all", audioPath] as const,

  bestSourceSegments: (audioPath: AudioPath) =>
    ["best-source-segments", audioPath] as const,

  synthesizeAll: () => ["synthesize"] as const,
  synthesizeStatus: (audioPath: AudioPath) =>
    ["synthesize", "status", audioPath] as const,
  synthesizeVoices: (audioPath: AudioPath) =>
    ["synthesize", "voices", audioPath] as const,
  synthesizeGenerated: (audioPath: AudioPath) =>
    ["synthesize", "generated", audioPath] as const,

  // ── Index & search ─────────────────────────────────────
  searchConfig: () => ["search", "config"] as const,
  searchStats: (showName: string) => ["search", "stats", showName] as const,
  indexedEpisodes: (showName: string, model: string, chunking: string) =>
    ["search", "indexed-episodes", showName, model, chunking] as const,
  indexedSpeakers: (showName: string, model: string, chunking: string) =>
    ["search", "indexed-speakers", showName, model, chunking] as const,
  indexedEpisode: (showName: string, stem: string, model: string, chunking: string) =>
    ["search", "indexed-episode", showName, stem, model, chunking] as const,

  indexConfig: () => ["index", "config"] as const,
  indexStatus: (audioPath: AudioPath, showName: string) =>
    ["index", "status", audioPath, showName] as const,
  /** Collections this episode currently lives in (one row per entry). */
  episodeCollections: (audioPath: AudioPath, showName: string) =>
    ["index", "episode-collections", audioPath, showName] as const,
  indexInspect: (
    audioPath: AudioPath,
    showName: string,
    model: string,
    chunking: string,
  ) => ["index", "inspect", audioPath, showName, model, chunking] as const,

  // ── Integrations & MCP ────────────────────────────────
  claudeDesktop: () => ["integrations", "claude-desktop"] as const,
  mcpPrompts: () => ["mcp", "prompts"] as const,

  // ── Bot access ────────────────────────────────────────
  showAccess: (showName: string) =>
    ["bot-access", "show", showName] as const,
  showAccessList: () => ["bot-access", "list"] as const,

  // ── Step-scoped (TranscriptViewer editor key: "transcribe" | "correct" | "translate-xxx") ──
  /** All queries for a given editor step. */
  stepAll: (editorKey: string) => [editorKey] as const,
  /** Segments for a step. */
  stepSegments: (editorKey: string, audioPath: AudioPath) =>
    [editorKey, "segments", audioPath] as const,
  /** Version list for a step. */
  stepVersions: (editorKey: string, audioPath: AudioPath) =>
    [editorKey, "versions", audioPath] as const,
  /** A specific version's segments. */
  stepVersionSegments: (
    editorKey: string,
    audioPath: AudioPath,
    versionId: string | null,
  ) => [editorKey, "versions", audioPath, versionId] as const,
};
