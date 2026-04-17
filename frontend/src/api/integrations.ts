import { json } from "./client";

export interface ClaudeDesktopStatus {
  enabled: boolean;
  config_path: string;
  command_path: string;
  claude_desktop_installed: boolean;
  mcp_available: boolean;
  needs_restart_hint: string;
}

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
