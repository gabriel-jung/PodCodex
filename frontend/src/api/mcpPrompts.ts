import { json, rawFetch } from "./client";
import type { PromptOut, SlotIn } from "./generated-types";

/** A prompt slot. `SlotIn` on the Python side; the create/update payloads
 *  below reuse it, which is why `type` and `required` stay writable. */
export type SlotDef = SlotIn;
export type SlotType = SlotIn["type"];
export type McpPrompt = PromptOut;

export interface McpPromptCreate {
  id: string;
  name?: string;
  title: string;
  description?: string;
  template: string;
  slots?: SlotDef[];
  enabled?: boolean;
}

export interface McpPromptUpdate {
  name?: string;
  title?: string;
  description?: string;
  template?: string;
  slots?: SlotDef[];
  enabled?: boolean;
}

const jsonHeaders = { "Content-Type": "application/json" };

export const getMcpPrompts = () => json<McpPrompt[]>("/api/mcp/prompts");

export const createMcpPrompt = (payload: McpPromptCreate) =>
  json<McpPrompt>("/api/mcp/prompts", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify(payload),
  });

export const updateMcpPrompt = (id: string, payload: McpPromptUpdate) =>
  json<McpPrompt>(`/api/mcp/prompts/${encodeURIComponent(id)}`, {
    method: "PUT",
    headers: jsonHeaders,
    body: JSON.stringify(payload),
  });

export async function deleteMcpPrompt(id: string): Promise<void> {
  await rawFetch(`/api/mcp/prompts/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export const toggleMcpPrompt = (id: string) =>
  json<McpPrompt>(`/api/mcp/prompts/${encodeURIComponent(id)}/toggle`, {
    method: "POST",
  });
