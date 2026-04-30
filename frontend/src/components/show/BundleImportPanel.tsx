import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { FolderOpen, Loader2, Package, AlertTriangle } from "lucide-react";

import { getConfig, importBundle, previewBundle } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import type { ImportRequest, ImportResult } from "@/api/generated-types";
import { Button } from "@/components/ui/button";
import FolderPicker from "@/components/common/FolderPicker";
import { errorMessage, parentPath, selectClass, splitPath } from "@/lib/utils";

type Policy = Exclude<NonNullable<ImportRequest["on_conflict"]>, "auto">;

interface Props {
  /** Called once import completes — host modal navigates to the show. */
  onImported: (showsDir: string, finalFolderName: string | null) => void;
}

export default function BundleImportPanel({ onImported }: Props) {
  const { data: config } = useQuery({
    queryKey: queryKeys.config(),
    queryFn: getConfig,
  });

  const [showsDir, setShowsDir] = useState("");
  const [nameOverride, setNameOverride] = useState("");
  const [policy, setPolicy] = useState<Policy | null>(null);
  const [archivePickerOpen, setArchivePickerOpen] = useState(false);
  const [showsDirPickerOpen, setShowsDirPickerOpen] = useState(false);

  const previewMutation = useMutation({
    mutationFn: (path: string) => previewBundle(path),
    onSuccess: (p) => {
      // Mode-appropriate default policy. User can override.
      setPolicy(p.manifest.mode === "full" ? "rename" : "replace");
      if (p.manifest.mode === "full") {
        const first = config?.show_folders?.[0];
        if (first) setShowsDir(parentPath(first));
      }
    },
  });
  const archivePath = previewMutation.variables ?? null;
  const preview = previewMutation.data ?? null;
  const previewError = previewMutation.error
    ? errorMessage(previewMutation.error)
    : null;

  const importMutation = useMutation({
    mutationFn: () => {
      if (!archivePath) throw new Error("no archive selected");
      return importBundle({
        archive_path: archivePath,
        shows_dir: preview?.manifest.mode === "full" ? showsDir : undefined,
        name: nameOverride.trim() || undefined,
        on_conflict: policy ?? "auto",
      });
    },
    onSuccess: (res: ImportResult) => {
      const finalFolder = res.shows_imported[0] ?? null;
      onImported(res.shows_dir, finalFolder);
    },
  });

  const onArchivePicked = (path: string) => {
    setArchivePickerOpen(false);
    setNameOverride("");
    previewMutation.mutate(path);
  };

  const isFull = preview?.manifest.mode === "full";
  const singleShow = preview && preview.manifest.shows.length === 1;
  const importDisabled =
    !preview ||
    importMutation.isPending ||
    (isFull && !showsDir.trim());

  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs text-muted-foreground">
        Import a <code className="font-mono">.podcodex</code> archive shared by another user or exported from this app. Bundles can contain a complete show or just its search index.
      </p>

      <Button
        onClick={() => setArchivePickerOpen(true)}
        variant="outline"
        className="justify-start gap-2"
        disabled={previewMutation.isPending || importMutation.isPending}
      >
        {previewMutation.isPending ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <FolderOpen className="w-4 h-4" />
        )}
        {archivePath ? splitPath(archivePath).basename : "Choose a .podcodex file…"}
      </Button>

      {previewError && (
        <p className="text-xs text-destructive">{previewError}</p>
      )}

      {preview && (
        <div className="rounded-lg border border-border bg-muted/30 p-3 space-y-2">
          <div className="flex items-center gap-2 text-sm">
            <Package className="w-4 h-4 text-muted-foreground" />
            <span className="font-medium">
              {preview.manifest.mode === "full" ? "Full bundle" : "Search index only"}
            </span>
            <span className="text-xs text-muted-foreground">
              ({(preview.size_bytes / 1e6).toFixed(2)} MB)
            </span>
          </div>

          {preview.manifest.shows.map((s, i) => (
            <div key={i} className="text-xs text-muted-foreground pl-6 space-y-0.5">
              <div>
                <span className="text-foreground font-medium">{s.name}</span>
                {s.audio_included && (
                  <span className="ml-1 text-[10px] text-muted-foreground">+ audio</span>
                )}
              </div>
              {s.collections.length > 0 && (
                <div className="font-mono text-[10px]">
                  {s.collections.map((c) => `${c.model}/${c.chunker} (${c.rows} rows)`).join(", ")}
                </div>
              )}
            </div>
          ))}

          {preview.embedder_warnings.length > 0 && (
            <div className="flex items-start gap-1.5 text-xs text-warning">
              <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              <ul className="list-none space-y-0.5">
                {preview.embedder_warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {preview && isFull && (
        <div className="space-y-2">
          <label className="text-xs text-muted-foreground block">
            Save shows to
          </label>
          <div className="flex gap-2">
            <input
              value={showsDir}
              onChange={(e) => setShowsDir(e.target.value)}
              placeholder="/path/to/shows/parent"
              className="input flex-1 font-mono text-xs"
            />
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setShowsDirPickerOpen(true)}
            >
              Browse
            </Button>
          </div>
          {singleShow && (
            <div>
              <label className="text-xs text-muted-foreground block mb-1">
                Folder name (optional override)
              </label>
              <input
                value={nameOverride}
                onChange={(e) => setNameOverride(e.target.value)}
                placeholder={preview.manifest.shows[0].folder}
                className="input w-full"
              />
            </div>
          )}
        </div>
      )}

      {preview && (
        <div>
          <label className="text-xs text-muted-foreground block mb-1">
            On conflict
          </label>
          <select
            value={policy ?? (isFull ? "rename" : "replace")}
            onChange={(e) => setPolicy(e.target.value as Policy)}
            className={selectClass}
          >
            {isFull && (
              <option value="rename">Rename folder, replace index</option>
            )}
            <option value="replace">Replace {isFull ? "folder + index" : "existing"}</option>
            <option value="abort">Abort on conflict</option>
          </select>
        </div>
      )}

      {importMutation.isError && (
        <p className="text-xs text-destructive">
          {errorMessage(importMutation.error)}
        </p>
      )}

      <div className="flex justify-end">
        <Button
          onClick={() => importMutation.mutate()}
          disabled={importDisabled}
          size="sm"
          className="gap-1.5"
        >
          {importMutation.isPending ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <Package className="w-3.5 h-3.5" />
          )}
          {importMutation.isPending ? "Importing…" : "Import"}
        </Button>
      </div>

      <FolderPicker
        open={archivePickerOpen}
        mode="file"
        extensions={["podcodex"]}
        onClose={() => setArchivePickerOpen(false)}
        onSelect={onArchivePicked}
        initialPath={archivePath ? parentPath(archivePath) : "~"}
        title="Choose a .podcodex archive"
      />

      <FolderPicker
        open={showsDirPickerOpen}
        onClose={() => setShowsDirPickerOpen(false)}
        onSelect={(p) => {
          setShowsDirPickerOpen(false);
          setShowsDir(p);
        }}
        initialPath={showsDir || "~"}
        title="Choose where to save the imported show"
      />
    </div>
  );
}
