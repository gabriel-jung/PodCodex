import { createContext, useContext } from "react";
import type { Platform } from "./types";
import { webPlatform } from "./web";

export const PlatformCtx = createContext<Platform>(webPlatform);

export function usePlatform(): Platform {
  return useContext(PlatformCtx);
}
