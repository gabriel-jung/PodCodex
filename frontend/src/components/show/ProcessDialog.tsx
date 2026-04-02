/** Quick-process dialog: preset picker + step toggles for batch pipeline. */

import { useEffect, useState } from "react";
import { usePipelineConfigStore, PIPELINE_PRESETS } from "@/stores/pipelineConfigStore";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Play } from "lucide-react";

interface ProcessDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onRun: (opts: {
    preset: string;
    transcribe: boolean;
    polish: boolean;
    translate: boolean;
    index: boolean;
  }) => void;
  disabled?: boolean;
  episodeCount: number;
}

export default function ProcessDialog({
  open,
  onOpenChange,
  onRun,
  disabled,
  episodeCount,
}: ProcessDialogProps) {
  const { preset, setPreset } = usePipelineConfigStore();
  const llmApiKey = usePipelineConfigStore((s) => s.llm.apiKey);

  const [transcribe, setTranscribe] = useState(true);
  const [polish, setPolish] = useState(false);
  const [translate, setTranslate] = useState(false);
  const [index, setIndex] = useState(true);

  // Reset step toggles to defaults when dialog opens
  useEffect(() => {
    if (open) {
      setTranscribe(true);
      setPolish(false);
      setTranslate(false);
      setIndex(true);
    }
  }, [open]);

  const hasApiKey = !!llmApiKey;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Process {episodeCount} episode{episodeCount !== 1 ? "s" : ""}</DialogTitle>
        </DialogHeader>

        {/* Preset selector */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Pipeline preset</label>
          <div className="grid grid-cols-3 gap-2">
            {Object.entries(PIPELINE_PRESETS).map(([key, p]) => (
              <button
                key={key}
                onClick={() => setPreset(key)}
                className={`rounded-lg border p-3 text-left transition ${
                  preset === key
                    ? "border-primary bg-accent"
                    : "border-border hover:border-primary/50"
                }`}
              >
                <div className="text-sm font-medium">{p.label}</div>
                <div className="text-[11px] text-muted-foreground mt-0.5">{p.desc}</div>
                <div className="text-[10px] text-muted-foreground mt-1 font-mono">
                  {p.whisperModel}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Step toggles */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Steps</label>
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={transcribe}
                onChange={(e) => setTranscribe(e.target.checked)}
                className="accent-primary"
              />
              Transcribe
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={polish}
                onChange={(e) => setPolish(e.target.checked)}
                className="accent-primary"
                disabled={!hasApiKey}
              />
              <span className={!hasApiKey ? "text-muted-foreground" : ""}>
                Polish
              </span>
              {!hasApiKey && (
                <span className="text-[11px] text-muted-foreground ml-1">
                  (requires API key in Settings)
                </span>
              )}
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={translate}
                onChange={(e) => setTranslate(e.target.checked)}
                className="accent-primary"
                disabled={!hasApiKey}
              />
              <span className={!hasApiKey ? "text-muted-foreground" : ""}>
                Translate
              </span>
              {!hasApiKey && (
                <span className="text-[11px] text-muted-foreground ml-1">
                  (requires API key in Settings)
                </span>
              )}
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={index}
                onChange={(e) => setIndex(e.target.checked)}
                className="accent-primary"
              />
              Index for search
            </label>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={() => {
              onRun({ preset, transcribe, polish, translate, index });
              onOpenChange(false);
            }}
            disabled={disabled || (!transcribe && !polish && !translate && !index)}
          >
            <Play className="w-3.5 h-3.5 mr-1" />
            Run
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
