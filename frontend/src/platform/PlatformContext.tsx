import { createContext, useContext } from "react";
import type { Platform } from "./types";
import { webPlatform } from "./web";

const PlatformCtx = createContext<Platform>(webPlatform);

export function PlatformProvider({ children }: { children: React.ReactNode }) {
  // For now, always use web platform.
  // When Tauri integration is needed, detect window.__TAURI__ and swap implementation.
  return (
    <PlatformCtx.Provider value={webPlatform}>
      {children}
    </PlatformCtx.Provider>
  );
}

export function usePlatform(): Platform {
  return useContext(PlatformCtx);
}
