import { useEffect, useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Download, Loader2 } from "lucide-react";

import { exportShowBundle } from "@/api/client";
import { Button } from "@/components/ui/button";
import { SettingSection } from "@/components/ui/setting-row";
import FolderPicker from "@/components/common/FolderPicker";
import { errorMessage, parentPath, splitPath } from "@/lib/utils";

interface Props {
  folder: string;
  showName: string;
}

const folderBasename = (path: string): string =>
  splitPath(path.replace(/[\\/]+$/, "")).basename || "show";

const slugify = (s: string, fallback: string): string => {
  const slug = s
    .replace(/[^a-zA-Z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase();
  return slug || fallback;
};

export default function BundleExportSection({ folder, showName }: Props) {
  const defaultSlug = useMemo(
    () => slugify(showName, folderBasename(folder)),
    [showName, folder],
  );
  const [saveDir, setSaveDir] = useState(parentPath(folder));
  const [filename, setFilename] = useState(`${defaultSlug}.podcodex`);
  const [withAudio, setWithAudio] = useState(false);
  const [resultMsg, setResultMsg] = useState<string | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);

  useEffect(() => {
    setFilename(`${defaultSlug}.podcodex`);
  }, [defaultSlug]);

  const exportMutation = useMutation({
    mutationFn: async (variant: "full" | "index") => {
      const cleanFilename =
        filename.trim() ||
        (variant === "full"
          ? `${defaultSlug}.podcodex`
          : `${defaultSlug}-index.podcodex`);
      const target = `${saveDir.replace(/\/+$/, "")}/${cleanFilename}`;
      return exportShowBundle({
        show_folder: folder,
        output_path: target,
        with_audio: variant === "full" ? withAudio : false,
        index_only: variant === "index",
      });
    },
    onMutate: () => setResultMsg("Exporting… this may take a moment for large shows."),
    onSuccess: (res) => {
      setResultMsg(
        `Exported ${(res.size_bytes / 1e6).toFixed(2)} MB → ${res.output_path}`,
      );
    },
    onError: (err) => setResultMsg(`Error: ${errorMessage(err)}`),
  });

  const busy = exportMutation.isPending;
  const runningVariant = busy ? exportMutation.variables : null;
  const canExport = saveDir.trim() && filename.trim() && !busy;

  return (
    <SettingSection
      title="Sharing"
      description="Bundle this show into a portable .podcodex archive — full content (transcripts + index) for sharing with another user, or index-only for selective bot deploy."
    >
      <div className="space-y-3">
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
            placeholder={`${defaultSlug}.podcodex`}
            className="input w-full"
            disabled={busy}
          />
        </div>

        <div className="flex items-center gap-2">
          <input
            id="bundle-with-audio"
            type="checkbox"
            checked={withAudio}
            onChange={(e) => setWithAudio(e.target.checked)}
            disabled={busy}
            className="accent-primary"
          />
          <label htmlFor="bundle-with-audio" className="text-sm cursor-pointer">
            Include audio files in full bundle
          </label>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button
            onClick={() => exportMutation.mutate("full")}
            disabled={!canExport}
            size="sm"
            variant="outline"
            className="gap-1.5"
          >
            {runningVariant === "full" ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Download className="w-3.5 h-3.5" />
            )}
            {runningVariant === "full" ? "Exporting…" : "Export full bundle"}
          </Button>
          <Button
            onClick={() => exportMutation.mutate("index")}
            disabled={!canExport}
            size="sm"
            variant="outline"
            className="gap-1.5"
          >
            {runningVariant === "index" ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Download className="w-3.5 h-3.5" />
            )}
            {runningVariant === "index" ? "Exporting…" : "Export search index only"}
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
      </div>

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
    </SettingSection>
  );
}
