/** Platform detection and export. */

export type { Platform, PlatformFS, PlatformWindow, PlatformLifecycle } from "./types";
export { PlatformProvider } from "./PlatformContext";
export { usePlatform } from "./context";
export { isTauri } from "./isTauri";
