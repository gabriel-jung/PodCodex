import { useQuery } from "@tanstack/react-query";
import { healthQueryOptions } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { isTauri } from "@/platform";

/**
 * Single facility for "what version am I running". Every version surface
 * (sidebar footer, boot splash, Settings > About) reads it from here.
 *
 * Two independent numbers, because they can drift:
 *
 * - `shell`: the Tauri desktop shell (`src-tauri/Cargo.toml` version).
 * - `backend`: the Python sidecar (`podcodex.__version__`), via /api/health.
 *
 * A normal install keeps them equal (`make bump` writes both). They diverge
 * in two quite different situations, and telling them apart matters because
 * the fixes are opposite:
 *
 * - **Backend newer than shell**: the app was updated while it was running.
 *   The window is still the old binary, and the sidecar it respawned came
 *   from the new bundle. This is the *normal* macOS upgrade path, where you
 *   drag the new app over the running one, so it is the common case by far.
 *   A restart fixes it; reinstalling does nothing.
 * - **Shell newer than backend**: a Windows MSI upgrade only half-applied
 *   (the WiX same-version-skip failure documented in CLAUDE.md), which is
 *   otherwise completely silent. Only a reinstall fixes that.
 *
 * `mismatch` only fires in bundle mode. In a dev checkout the backend version
 * comes from the editable install's dist-info, which lags `pyproject.toml`
 * until someone reinstalls, and falls back to `0.0.0+unknown` when the package
 * has no metadata at all. Both are normal in dev, and neither is fixed by the
 * "reinstall the app" advice the warning gives.
 *
 * `shell` resolves without the backend, so the boot splash can show a version
 * while the sidecar is still extracting. It deliberately does not go through
 * the `usePlatform()` context: the splash renders before `PlatformProvider`
 * mounts.
 */

/** Compare two dotted versions. >0 when `a` is newer, 0 when equal or either
 *  is unparseable (an unknown ordering must not claim a direction). */
export function compareVersions(a: string, b: string): number {
  const parse = (v: string) => {
    const core = v.split(/[-+]/, 1)[0];
    const parts = core.split(".").map((n) => Number.parseInt(n, 10));
    return parts.length === 3 && parts.every(Number.isFinite) ? parts : null;
  };
  const pa = parse(a);
  const pb = parse(b);
  if (!pa || !pb) return 0;
  for (let i = 0; i < 3; i++) {
    if (pa[i] !== pb[i]) return pa[i] > pb[i] ? 1 : -1;
  }
  return 0;
}

async function fetchShellVersion(): Promise<string | null> {
  if (!isTauri()) return null;
  try {
    const { getVersion } = await import("@tauri-apps/api/app");
    return await getVersion();
  } catch {
    // Running in a browser against `make dev-api`, or the app plugin is
    // unavailable. Not an error worth surfacing; the backend version stands
    // on its own.
    return null;
  }
}

export interface Versions {
  /** Tauri shell version, or null on web / when unavailable. */
  shell: string | null;
  /** Backend sidecar version, or null until /api/health resolves. */
  backend: string | null;
  /** Both known, different, and running a packaged build. Never set in dev,
   *  where drift is routine and harmless. */
  mismatch: boolean;
  /** A mismatch whose backend is *newer*: the app was updated while running
   *  and is waiting on a restart, not on a reinstall. */
  needsRestart: boolean;
  /** Best single version to show when there is only room for one. */
  display: string | null;
}

export function useVersions(): Versions {
  const { data: shell } = useQuery({
    queryKey: queryKeys.shellVersion(),
    queryFn: fetchShellVersion,
    staleTime: Infinity,
    gcTime: Infinity,
    refetchOnWindowFocus: false,
    retry: false,
  });

  const { data: health } = useQuery(healthQueryOptions);

  const shellVersion = shell ?? null;
  const backend = health?.version ?? null;
  const isBundle = health?.mode === "bundle";
  const mismatch =
    isBundle && !!shellVersion && !!backend && shellVersion !== backend;

  return {
    shell: shellVersion,
    backend,
    mismatch,
    needsRestart: mismatch && compareVersions(backend!, shellVersion!) > 0,
    // Prefer the shell: it is what the installer wrote and what the user sees
    // in Windows "Apps & features", so it matches their mental model.
    display: shellVersion ?? backend,
  };
}
