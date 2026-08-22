import { useQuery } from "@tanstack/react-query";
import { AlertCircle, Copy, Info } from "lucide-react";
import { getAbout } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { Button } from "@/components/ui/button";
import { SettingRow } from "@/components/ui/setting-row";
import { useVersions } from "@/hooks/useVersions";
import { useState } from "react";

/**
 * Version and environment readout.
 *
 * Windows ships without an app menu, so this is the only place a user can
 * answer "which version am I on" before something breaks. The copy button
 * exists so bug reports arrive with the environment already attached.
 */
export default function AboutPanel() {
  const { shell, backend, mismatch, needsRestart } = useVersions();
  const { data: about } = useQuery({
    queryKey: queryKeys.about(),
    queryFn: getAbout,
    staleTime: Infinity,
  });

  const rows: { label: string; value: string; help?: string }[] = [];
  if (shell) {
    rows.push({
      label: "App version",
      value: shell,
      help: "Desktop shell, as reported by the installer",
    });
  }
  if (backend) {
    rows.push({
      label: shell ? "Backend version" : "Version",
      value: backend,
      help: shell ? "Transcription server bundled with the app" : undefined,
    });
  }
  if (about) {
    rows.push({ label: "Platform", value: `${about.platform} (${about.machine})` });
    rows.push({ label: "Python", value: about.python_version });
    if (about.mode === "dev") {
      // Bundle is the norm for everyone running an installer; only the
      // developer-mode deviation is worth a row.
      rows.push({
        label: "Mode",
        value: "dev",
        help: "Running from a venv, not the packaged sidecar",
      });
    }
    rows.push({ label: "Data folder", value: about.data_dir });
    rows.push({ label: "Log file", value: about.log_path });
  }

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold flex items-center gap-2">
          <Info className="w-4 h-4" /> About
        </h2>
        <CopyDiagnostics shell={shell} backend={backend} rows={rows} />
      </div>

      {mismatch && (
        <div className="rounded-lg border border-warning/40 bg-warning/5 p-4 text-sm space-y-1">
          <div className="flex items-center gap-2 font-medium text-warning">
            <AlertCircle className="w-4 h-4" />
            {needsRestart ? "Restart to finish updating" : "Incomplete update"}
          </div>
          {needsRestart ? (
            <p className="text-xs text-muted-foreground">
              Version <code className="font-mono text-2xs">{backend}</code> is installed,
              but this window is still running{" "}
              <code className="font-mono text-2xs">{shell}</code> because the app was
              open while it updated. Quit PodCodex and open it again. There is no
              need to reinstall.
            </p>
          ) : (
            <p className="text-xs text-muted-foreground">
              The app is version <code className="font-mono text-2xs">{shell}</code> but
              its backend is <code className="font-mono text-2xs">{backend}</code>. An
              installer run replaced only part of the app. Reinstall the latest
              version to get the two back in step.
            </p>
          )}
        </div>
      )}

      <div className="border border-border rounded-lg px-4 divide-y divide-border/40">
        {rows.map((row) => (
          <SettingRow key={row.label} label={row.label} help={row.help}>
            <span className="font-mono text-xs text-muted-foreground break-all">
              {row.value}
            </span>
          </SettingRow>
        ))}
      </div>
    </section>
  );
}

function CopyDiagnostics({
  shell,
  backend,
  rows,
}: {
  shell: string | null;
  backend: string | null;
  rows: { label: string; value: string }[];
}) {
  const [copied, setCopied] = useState(false);
  if (!shell && !backend) return null;

  const onCopy = async () => {
    const text = rows.map((r) => `${r.label}: ${r.value}`).join("\n");
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard denied; nothing actionable to surface */
    }
  };

  return (
    <Button variant="outline" size="sm" onClick={onCopy}>
      <Copy className="w-4 h-4 mr-2" />
      {copied ? "Copied" : "Copy details"}
    </Button>
  );
}
