/**
 * Barrel re-export — all API functions available from "@/api/client".
 *
 * The actual implementations live in feature modules:
 *   health.ts, shows.ts, transcribe.ts, polish.ts, translate.ts,
 *   synthesize.ts, search.ts, filesystem.ts
 */

export * from "./health";
export * from "./shows";
export * from "./transcribe";
export * from "./polish";
export * from "./translate";
export * from "./synthesize";
export * from "./search";
export * from "./filesystem";
