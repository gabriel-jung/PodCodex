import { useEffect, useState } from "react";
import type { Platform } from "./types";
import { webPlatform } from "./web";
import { isTauri } from "./isTauri";
import { PlatformCtx } from "./context";

export function PlatformProvider({ children }: { children: React.ReactNode }) {
  const [platform, setPlatform] = useState<Platform>(webPlatform);

  useEffect(() => {
    if (isTauri()) {
      // Lazy-load Tauri module only when running inside Tauri
      import("./tauri").then((m) => setPlatform(m.tauriPlatform));
    }
  }, []);

  return (
    <PlatformCtx.Provider value={platform}>
      {children}
    </PlatformCtx.Provider>
  );
}
