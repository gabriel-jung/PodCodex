import { useMutation, useQuery } from "@tanstack/react-query";
import { Download, Loader2, Package } from "lucide-react";
import { useState } from "react";

import { exportIndexBundle, listShows } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { Button } from "@/components/ui/button";
import FolderPicker from "@/components/common/FolderPicker";
import { errorMessage, parentPath } from "@/lib/utils";

const DEFAULT_FILENAME = "shows-index.podcodex";

export default function BundleExportPanel() {
  const [resultMsg, setResultMsg] = useState<string | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [filename, setFilename] = useState(DEFAULT_FILENAME);

  const { data: shows } = useQuery({
    queryKey: queryKeys.shows(),
    queryFn: listShows,
  });

  const initialSaveDir =
    shows && shows.length > 0 ? parentPath(shows[0].path) : "~";
  const [saveDir, setSaveDir] = useState(initialSaveDir);

  const exportMutation = useMutation({
    mutationFn: () => {
      const folders = (shows ?? []).map((s) => s.path);
      if (folders.length === 0) throw new Error("No registered shows to export");
      const cleanFilename = filename.trim() || DEFAULT_FILENAME;
      const target = `${saveDir.replace(/\/+$/, "")}/${cleanFilename}`;
      return exportIndexBundle({
        show_folders: folders,
        output_path: target,
      });
    },
    onMutate: () =>
      setResultMsg("Exporting all shows… this may take a moment."),
    onSuccess: (res) => {
      setResultMsg(
        `Exported ${res.shows_exported} show(s), ${res.collections_exported} collection(s) — ${(res.size_bytes / 1e6).toFixed(2)} MB → ${res.output_path}`,
      );
    },
    onError: (err) => setResultMsg(`Error: ${errorMessage(err)}`),
  });

  const showCount = shows?.length ?? 0;
  const busy = exportMutation.isPending;
  const canExport = saveDir.trim() && filename.trim() && !busy && showCount > 0;

  return (
    <section className="space-y-3 pt-6 border-t border-border">
      <h2 className="text-lg font-semibold flex items-center gap-2">
        <Package className="w-5 h-5" /> Index export
      </h2>
      <p className="text-sm text-muted-foreground">
        Bundle every show&apos;s search index into a single <code className="font-mono">.podcodex</code> file. Useful for deploying the Discord bot to a VPS without rsync — copy the file across, then run <code className="font-mono">podcodex-import</code> on the host.
      </p>

      <div>
        <label className="text-xs text-muted-foreground block mb-1">
          Save location
        </label>
        <div className="flex gap-2">
          <input
            value={saveDir}
            onChange={(e) => setSaveDir(e.target.value)}
            className="input flex-1 text-xs font-mono"
            disabled={busy}
          />
          <Button
            onClick={() => setPickerOpen(true)}
            variant="outline"
            size="sm"
            disabled={busy}
          >
            Browse…
          </Button>
        </div>
      </div>

      <div>
        <label className="text-xs text-muted-foreground block mb-1">
          Filename
        </label>
        <input
          value={filename}
          onChange={(e) => setFilename(e.target.value)}
          placeholder={DEFAULT_FILENAME}
          className="input w-full"
          disabled={busy}
        />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <Button
          onClick={() => exportMutation.mutate()}
          disabled={!canExport}
          size="sm"
          variant="outline"
          className="gap-1.5"
        >
          {busy ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <Download className="w-3.5 h-3.5" />
          )}
          {busy ? "Exporting…" : `Export all shows (${showCount})`}
        </Button>
      </div>

      {resultMsg && (
        <p
          className={`text-xs ${
            resultMsg.startsWith("Error")
              ? "text-destructive"
              : "text-muted-foreground"
          }`}
        >
          {resultMsg}
        </p>
      )}

      <FolderPicker
        open={pickerOpen}
        onClose={() => setPickerOpen(false)}
        onSelect={(p) => {
          setSaveDir(p);
          setPickerOpen(false);
        }}
        initialPath={saveDir}
        title="Choose a save location"
      />
    </section>
  );
}
