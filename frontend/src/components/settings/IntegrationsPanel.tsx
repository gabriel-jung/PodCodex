import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { ArrowLeft, MessageSquare, Plug } from "lucide-react";

import { getClaudeDesktopStatus } from "@/api/integrations";
import { getShowAccessList } from "@/api/botAccess";
import { queryKeys } from "@/api/queryKeys";
import ClaudeDesktopPanel from "./ClaudeDesktopPanel";
import { IntegrationCard } from "./IntegrationCard";
import { StatusDot } from "@/components/ui/status-dot";

type View = "gallery" | "claude-desktop" | "discord";

export default function IntegrationsPanel() {
  const [view, setView] = useState<View>("gallery");

  if (view === "claude-desktop") {
    return (
      <section className="space-y-6">
        <BackLink onClick={() => setView("gallery")} />
        <ClaudeDesktopPanel />
      </section>
    );
  }

  if (view === "discord") {
    return (
      <section className="space-y-6">
        <BackLink onClick={() => setView("gallery")} />
        <DiscordOverview />
      </section>
    );
  }

  return <Gallery onOpen={setView} />;
}


function Gallery({ onOpen }: { onOpen: (view: View) => void }) {
  const { data: claudeStatus } = useQuery({
    queryKey: queryKeys.claudeDesktop(),
    queryFn: getClaudeDesktopStatus,
  });
  const { data: accessList } = useQuery({
    queryKey: queryKeys.showAccessList(),
    queryFn: getShowAccessList,
    retry: false,
  });

  const protectedCount = accessList?.filter((a) => a.is_protected).length ?? 0;
  const discordState = protectedCount > 0 ? "ok" : "idle";
  const discordLabel =
    protectedCount > 0
      ? `${protectedCount} protected ${protectedCount === 1 ? "show" : "shows"}`
      : "Public access";

  const claudeState = claudeStatus?.mcp_available === false
    ? "warn"
    : claudeStatus?.enabled
    ? "ok"
    : "idle";
  const claudeLabel = claudeStatus?.mcp_available === false
    ? "Extra missing"
    : claudeStatus?.enabled
    ? "Connected"
    : "Off";

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <h2 className="text-lg font-semibold">Integrations</h2>
        <p className="text-sm text-muted-foreground">
          Connect PodCodex to tools that consume your podcast data.
        </p>
      </header>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <IntegrationCard
          icon={Plug}
          title="Claude Desktop"
          description="Search and query your indexed podcasts directly from Claude Desktop via the Model Context Protocol."
          statusState={claudeState}
          statusLabel={claudeLabel}
          onOpen={() => onOpen("claude-desktop")}
        />
        <IntegrationCard
          icon={MessageSquare}
          title="Discord bot"
          description="Query your podcasts from any Discord server. Protect individual shows with a password."
          statusState={discordState}
          statusLabel={discordLabel}
          onOpen={() => onOpen("discord")}
        />
      </div>
    </div>
  );
}


function BackLink({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition"
    >
      <ArrowLeft className="w-3.5 h-3.5" />
      Integrations
    </button>
  );
}


function DiscordOverview() {
  const { data: accessList, isLoading } = useQuery({
    queryKey: queryKeys.showAccessList(),
    queryFn: getShowAccessList,
    retry: false,
  });

  return (
    <div className="space-y-6">
      <header className="flex items-center gap-2">
        <MessageSquare size={18} className="text-muted-foreground" />
        <h3 className="text-base font-semibold">Discord bot</h3>
      </header>

      <div className="rounded-md border border-border bg-muted/30 p-4 text-sm text-muted-foreground space-y-2">
        <p>
          The Discord bot exposes your shows to any server that runs it. By
          default shows are public to anyone querying your bot. Set a password
          per show to require <code className="px-1 py-0.5 rounded bg-background">/unlock</code> before access.
        </p>
        <p>
          Access is managed per show. Open a show's Access section to generate
          or rotate its password.
        </p>
      </div>

      <div className="rounded-md border border-border overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-muted/40 text-xs text-muted-foreground">
            <tr>
              <th className="text-left font-medium px-3 py-2">Show</th>
              <th className="text-left font-medium px-3 py-2">Access</th>
            </tr>
          </thead>
          <tbody>
            {isLoading && (
              <tr>
                <td colSpan={2} className="px-3 py-4 text-center text-xs text-muted-foreground">
                  Loading…
                </td>
              </tr>
            )}
            {!isLoading && accessList?.length === 0 && (
              <tr>
                <td colSpan={2} className="px-3 py-4 text-center text-xs text-muted-foreground">
                  No shows yet.
                </td>
              </tr>
            )}
            {accessList?.map((a) => (
              <tr key={a.show} className="border-t border-border">
                <td className="px-3 py-2">{a.show}</td>
                <td className="px-3 py-2">
                  <span className="inline-flex items-center gap-1.5 text-xs">
                    <StatusDot state={a.is_protected ? "ok" : "idle"} />
                    {a.is_protected ? "Password-protected" : "Public"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
