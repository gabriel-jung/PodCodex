import { BASE, json } from "./client";

export type SlotType = "string" | "enum" | "int" | "bool";

export interface SlotDef {
  name: string;
  type?: SlotType;
  required?: boolean;
  default?: string | null;
  options?: string[];
}

export interface McpPrompt {
  id: string;
  name: string;
  title: string;
  description: string;
  template: string;
  slots: SlotDef[];
  enabled: boolean;
  is_builtin: boolean;
}

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
  const res = await fetch(
    `${BASE}/api/mcp/prompts/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );
  if (!res.ok) {
    throw new Error(`${res.status}: ${await res.text()}`);
  }
}

export const toggleMcpPrompt = (id: string) =>
  json<McpPrompt>(`/api/mcp/prompts/${encodeURIComponent(id)}/toggle`, {
    method: "POST",
  });
