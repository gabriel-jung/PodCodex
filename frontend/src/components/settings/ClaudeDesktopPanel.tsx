import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  ChevronRight,
  Copy,
  Loader2,
  Lock,
  Pencil,
  Plug,
  Plus,
  RefreshCcw,
  Sparkles,
  Trash2,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import {
  disableClaudeDesktop,
  enableClaudeDesktop,
  getClaudeDesktopStatus,
} from "@/api/integrations";
import type { ClaudeDesktopStatus } from "@/api/integrations";
import {
  createMcpPrompt,
  deleteMcpPrompt,
  getMcpPrompts,
  toggleMcpPrompt,
  updateMcpPrompt,
} from "@/api/mcpPrompts";
import type {
  McpPrompt,
  McpPromptCreate,
  McpPromptUpdate,
} from "@/api/mcpPrompts";
import { queryKeys } from "@/api/queryKeys";
import PromptEditorModal from "@/components/settings/PromptEditorModal";
import { Button } from "@/components/ui/button";
import { confirmDialog } from "@/components/ui/confirm-dialog";
import { SettingRow, SettingSection } from "@/components/ui/setting-row";

const HOW_IT_WORKS_KEY = "podcodex.claudeDesktopHowItWorksSeen";


export default function ClaudeDesktopPanel() {
  const qc = useQueryClient();
  const { data: status, isLoading } = useQuery({
    queryKey: queryKeys.claudeDesktop(),
    queryFn: getClaudeDesktopStatus,
  });

  const [justChanged, setJustChanged] = useState(false);
  useEffect(() => {
    if (!justChanged) return;
    const t = setTimeout(() => setJustChanged(false), 30_000);
    return () => clearTimeout(t);
  }, [justChanged]);

  const toggleMutation = useMutation({
    mutationFn: (next: boolean) =>
      next ? enableClaudeDesktop() : disableClaudeDesktop(),
    onSuccess: (updated) => {
      qc.setQueryData(queryKeys.claudeDesktop(), updated);
      setJustChanged(true);
    },
  });

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <div className="flex items-center gap-2">
          <Plug size={18} className="text-muted-foreground" />
          <h3 className="text-base font-semibold">Claude Desktop (MCP)</h3>
        </div>
        <p className="text-sm text-muted-foreground">
          Model Context Protocol lets Claude Desktop call PodCodex as a tool.
          Ask Claude to search your podcasts and it will query PodCodex directly.
        </p>
      </header>

      <HowItWorks />

      <SettingSection title="Integration">
        <SettingRow
          label="Enable"
          help="Register PodCodex as an MCP server so Claude Desktop can search your indexed podcasts."
          below={
            justChanged && (
              <RestartBanner hint={status?.needs_restart_hint} />
            )
          }
        >
          <ToggleControl
            status={status}
            isLoading={isLoading}
            isToggling={toggleMutation.isPending}
            onToggle={(next) => toggleMutation.mutate(next)}
          />
        </SettingRow>

        <SettingRow
          label="Status"
          help="Live state, refreshed from the server."
        >
          <StatusChip status={status} isLoading={isLoading} />
        </SettingRow>

        <SettingRow
          label="Command"
          help="Claude Desktop will spawn this binary as a subprocess on startup."
        >
          <CopyField value={status?.command_path ?? ""} />
        </SettingRow>

        <SettingRow
          label="Config file"
          help="The Claude Desktop config PodCodex writes to."
        >
          <CopyField value={status?.config_path ?? ""} />
        </SettingRow>
      </SettingSection>

      {status?.mcp_available === false && (
        <p className="text-xs text-amber-600 dark:text-amber-400">
          The MCP extra is not installed on the backend. Run
          {" "}<code className="px-1 py-0.5 rounded bg-muted">uv sync --extra desktop</code>
          {" "}and restart PodCodex.
        </p>
      )}

      <PromptsSection
        onMutationSucceeded={() => setJustChanged(true)}
      />
    </div>
  );
}


function PromptsSection({
  onMutationSucceeded,
}: {
  onMutationSucceeded: () => void;
}) {
  const qc = useQueryClient();
  const { data: prompts, isLoading } = useQuery({
    queryKey: queryKeys.mcpPrompts(),
    queryFn: getMcpPrompts,
  });
  const [editorOpen, setEditorOpen] = useState(false);
  const [editing, setEditing] = useState<McpPrompt | null>(null);

  const existingIds = useMemo(() => prompts?.map((p) => p.id) ?? [], [prompts]);

  function invalidate() {
    qc.invalidateQueries({ queryKey: queryKeys.mcpPrompts() });
    onMutationSucceeded();
  }

  const toggleMut = useMutation({
    mutationFn: (id: string) => toggleMcpPrompt(id),
    onSuccess: invalidate,
  });
  const deleteMut = useMutation({
    mutationFn: (id: string) => deleteMcpPrompt(id),
    onSuccess: invalidate,
  });

  async function submitPrompt(payload: McpPromptCreate | McpPromptUpdate) {
    if (editing) {
      await updateMcpPrompt(editing.id, payload as McpPromptUpdate);
    } else {
      await createMcpPrompt(payload as McpPromptCreate);
    }
    invalidate();
  }

  return (
    <SettingSection
      title="Slash commands"
      description="Named templates that appear in Claude Desktop's / menu. The user invokes them explicitly."
    >
      <div className="py-3 space-y-3">
        <div className="flex items-center justify-between">
          <p className="text-xs text-muted-foreground">
            {isLoading
              ? "Loading..."
              : `${prompts?.length ?? 0} prompt${prompts?.length === 1 ? "" : "s"}`}
          </p>
          <Button
            size="sm"
            onClick={() => {
              setEditing(null);
              setEditorOpen(true);
            }}
            className="h-7 text-xs gap-1"
          >
            <Plus size={12} /> Add prompt
          </Button>
        </div>

        <div className="rounded-md border border-border overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-muted/40 text-xs text-muted-foreground">
              <tr>
                <th className="text-left font-medium px-3 py-2">id</th>
                <th className="text-left font-medium px-3 py-2">title</th>
                <th className="text-right font-medium px-3 py-2">slots</th>
                <th className="text-center font-medium px-3 py-2">enabled</th>
                <th className="text-right font-medium px-3 py-2 w-20">actions</th>
              </tr>
            </thead>
            <tbody>
              {(prompts ?? []).map((p) => (
                <tr key={p.id} className="border-t border-border">
                  <td className="px-3 py-2 font-mono text-xs">
                    <span className="inline-flex items-center gap-1">
                      {p.is_builtin && (
                        <Lock size={10} className="text-muted-foreground" aria-label="built-in" />
                      )}
                      {p.id}
                    </span>
                  </td>
                  <td className="px-3 py-2">{p.title}</td>
                  <td className="px-3 py-2 text-right text-muted-foreground text-xs">
                    {p.slots.length}
                  </td>
                  <td className="px-3 py-2 text-center">
                    <input
                      type="checkbox"
                      checked={p.enabled}
                      onChange={() => toggleMut.mutate(p.id)}
                      disabled={toggleMut.isPending}
                    />
                  </td>
                  <td className="px-3 py-2">
                    <div className="flex items-center justify-end gap-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0"
                        onClick={() => {
                          setEditing(p);
                          setEditorOpen(true);
                        }}
                        title="Edit"
                      >
                        <Pencil size={12} />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0"
                        disabled={p.is_builtin || deleteMut.isPending}
                        onClick={() => {
                          confirmDialog.open({
                            title: `Delete prompt "${p.id}"?`,
                            confirmLabel: "Delete",
                            variant: "destructive",
                            onConfirm: () => deleteMut.mutate(p.id),
                          });
                        }}
                        title={p.is_builtin ? "Built-ins cannot be deleted" : "Delete"}
                      >
                        <Trash2
                          size={12}
                          className={p.is_builtin ? "opacity-40" : ""}
                        />
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
              {!isLoading && prompts?.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-3 py-4 text-center text-xs text-muted-foreground">
                    No prompts yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <PromptEditorModal
        open={editorOpen}
        editing={editing}
        onClose={() => setEditorOpen(false)}
        onSubmit={submitPrompt}
        existingIds={existingIds}
      />
    </SettingSection>
  );
}


function HowItWorks() {
  const [open, setOpen] = useState(() => {
    try {
      return !localStorage.getItem(HOW_IT_WORKS_KEY);
    } catch {
      return true;
    }
  });

  function toggle() {
    setOpen((prev) => {
      const next = !prev;
      if (!next) {
        try {
          localStorage.setItem(HOW_IT_WORKS_KEY, "seen");
        } catch {
          /* ignore */
        }
      }
      return next;
    });
  }

  return (
    <div className="rounded-md border border-border bg-muted/40">
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        className="w-full flex items-center gap-2 px-3 py-2 text-sm text-left hover:bg-muted/60 rounded-md"
      >
        <Sparkles size={14} className="text-muted-foreground" />
        <span className="font-medium">How it works</span>
        <ChevronRight
          className={`ml-auto w-4 h-4 text-muted-foreground transition-transform ${open ? "rotate-90" : ""}`}
        />
      </button>
      {open && (
        <ol className="px-4 pb-3 pt-1 text-sm text-muted-foreground space-y-1.5 list-decimal list-inside">
          <li>Turn on the integration below.</li>
          <li>PodCodex writes an <code>mcpServers</code> entry into Claude Desktop's config.</li>
          <li>
            Fully quit Claude Desktop (Cmd+Q on macOS, Alt+F4 elsewhere) and
            reopen it. Claude Desktop reads its MCP config only at startup.
          </li>
          <li>
            Click the plug icon inside a Claude conversation. You'll see
            {" "}<code>podcodex</code> with its tools and prompts.
          </li>
          <li>
            Leave PodCodex running while you use Claude. Toggling off removes
            the entry.
          </li>
        </ol>
      )}
    </div>
  );
}


function ToggleControl({
  status,
  isLoading,
  isToggling,
  onToggle,
}: {
  status: ClaudeDesktopStatus | undefined;
  isLoading: boolean;
  isToggling: boolean;
  onToggle: (next: boolean) => void;
}) {
  const enabled = !!status?.enabled;
  const disabled = isLoading || isToggling || status?.mcp_available === false;
  return (
    <button
      type="button"
      role="switch"
      aria-checked={enabled}
      disabled={disabled}
      onClick={() => onToggle(!enabled)}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
        enabled ? "bg-primary" : "bg-input"
      } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
    >
      <span
        className={`inline-block h-5 w-5 transform rounded-full bg-background shadow transition-transform ${
          enabled ? "translate-x-5" : "translate-x-0.5"
        }`}
      />
    </button>
  );
}


function StatusChip({
  status,
  isLoading,
}: {
  status: ClaudeDesktopStatus | undefined;
  isLoading: boolean;
}) {
  if (isLoading) {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
        <Loader2 size={12} className="animate-spin" /> Loading
      </span>
    );
  }
  if (!status) return null;
  if (status.mcp_available === false) {
    return (
      <Chip color="amber">MCP extra not installed</Chip>
    );
  }
  if (status.enabled) {
    return <Chip color="green">Connected</Chip>;
  }
  if (!status.claude_desktop_installed) {
    return <Chip color="amber">Claude Desktop not detected</Chip>;
  }
  return <Chip color="muted">Off</Chip>;
}


function Chip({
  color,
  children,
}: {
  color: "green" | "amber" | "muted";
  children: React.ReactNode;
}) {
  const map = {
    green:
      "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-500/30",
    amber:
      "bg-amber-500/15 text-amber-700 dark:text-amber-300 border-amber-500/30",
    muted: "bg-muted text-muted-foreground border-border",
  } as const;
  return (
    <span
      className={`inline-flex items-center text-xs border rounded-full px-2 py-0.5 ${map[color]}`}
    >
      {children}
    </span>
  );
}


function CopyField({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const t = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(t);
  }, [copied]);
  async function doCopy() {
    if (!value) return;
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
    } catch {
      /* ignore */
    }
  }
  return (
    <div className="flex items-center gap-1.5">
      <code className="text-xs px-2 py-1 rounded bg-muted max-w-[22rem] truncate">
        {value || "—"}
      </code>
      <Button
        variant="ghost"
        size="sm"
        onClick={doCopy}
        disabled={!value}
        className="h-7 w-7 p-0"
        title="Copy"
        aria-label={`Copy ${value}`}
      >
        {copied ? <Check size={14} /> : <Copy size={14} />}
      </Button>
    </div>
  );
}


function RestartBanner({ hint }: { hint?: string }) {
  return (
    <div className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-800 dark:text-amber-200">
      <RefreshCcw size={14} className="mt-0.5 shrink-0" />
      <span>{hint ?? "Restart Claude Desktop to apply this change."}</span>
    </div>
  );
}
