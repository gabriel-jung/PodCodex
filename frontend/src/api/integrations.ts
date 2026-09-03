import { json } from "./client";
import type { ClaudeDesktopStatus } from "./generated-types";

export type { ClaudeDesktopStatus };

export const getClaudeDesktopStatus = () =>
  json<ClaudeDesktopStatus>("/api/integrations/claude-desktop");

export const enableClaudeDesktop = () =>
  json<ClaudeDesktopStatus>("/api/integrations/claude-desktop/enable", {
    method: "POST",
  });

export const disableClaudeDesktop = () =>
  json<ClaudeDesktopStatus>("/api/integrations/claude-desktop/disable", {
    method: "POST",
  });
