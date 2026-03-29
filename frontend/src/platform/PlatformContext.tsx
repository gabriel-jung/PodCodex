import { createContext, useContext, useEffect, useState } from "react";
import type { Platform } from "./types";
import { webPlatform } from "./web";

const PlatformCtx = createContext<Platform>(webPlatform);

export function PlatformProvider({ children }: { children: React.ReactNode }) {
  const [platform, setPlatform] = useState<Platform>(webPlatform);

  useEffect(() => {
    if ((window as any).__TAURI__) {
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

export function usePlatform(): Platform {
  return useContext(PlatformCtx);
}
