import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Copy, Check } from "lucide-react";

interface ManualModePanelProps {
  generatePrompts: () => Promise<PromptBatch[]>;
  applyCorrections: (corrections: unknown[]) => Promise<unknown>;
  onApplied?: () => void;
}

interface PromptBatch {
  batch_index: number;
  prompt: string;
  segment_count: number;
}

export default function ManualModePanel({
  generatePrompts,
  applyCorrections,
  onApplied,
}: ManualModePanelProps) {
  const [prompts, setPrompts] = useState<PromptBatch[] | null>(null);
  const [corrections, setCorrections] = useState("");
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);

  const generateMutation = useMutation({
    mutationFn: generatePrompts,
    onSuccess: (data) => setPrompts(data),
  });

  const applyMutation = useMutation({
    mutationFn: () => {
      const parsed = JSON.parse(corrections);
      const arr = Array.isArray(parsed) ? parsed : [parsed];
      return applyCorrections(arr);
    },
    onSuccess: () => {
      setCorrections("");
      setPrompts(null);
      onApplied?.();
    },
  });

  const copyToClipboard = async (text: string, idx: number) => {
    await navigator.clipboard.writeText(text);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(null), 2000);
  };

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">
        Generate prompts, paste them into your LLM of choice, then paste the JSON response back.
      </p>

      {/* Generate */}
      {!prompts && (
        <Button
          onClick={() => generateMutation.mutate()}
          disabled={generateMutation.isPending}
          variant="outline"
          size="sm"
        >
          {generateMutation.isPending ? "Generating..." : "Generate prompts"}
        </Button>
      )}
      {generateMutation.isError && (
        <p className="text-destructive text-xs">{(generateMutation.error as Error).message}</p>
      )}

      {/* Prompt batches */}
      {prompts && (
        <div className="space-y-3">
          <p className="text-xs text-muted-foreground">
            {prompts.length} batch{prompts.length !== 1 ? "es" : ""} generated.
            Copy each prompt, run it in your LLM, then paste the combined JSON below.
          </p>
          <div className="max-h-64 overflow-y-auto space-y-2">
            {prompts.map((batch) => (
              <div key={batch.batch_index} className="border border-border rounded">
                <div className="flex items-center justify-between px-3 py-1.5 bg-secondary/50 border-b border-border">
                  <span className="text-xs text-muted-foreground">
                    Batch {batch.batch_index + 1} ({batch.segment_count} segments)
                  </span>
                  <Button
                    onClick={() => copyToClipboard(batch.prompt, batch.batch_index)}
                    variant="ghost"
                    size="sm"
                    className="h-6 px-2"
                  >
                    {copiedIdx === batch.batch_index ? (
                      <Check className="w-3 h-3 text-green-400" />
                    ) : (
                      <Copy className="w-3 h-3" />
                    )}
                  </Button>
                </div>
                <pre className="p-2 text-xs max-h-32 overflow-y-auto whitespace-pre-wrap">
                  {batch.prompt}
                </pre>
              </div>
            ))}
          </div>

          {/* Apply corrections */}
          <div className="space-y-2">
            <label className="text-xs text-muted-foreground block">
              Paste LLM JSON response:
            </label>
            <textarea
              value={corrections}
              onChange={(e) => setCorrections(e.target.value)}
              placeholder='[{"speaker": "...", "text": "...", "start": 0, "end": 0}, ...]'
              className="input text-xs w-full"
              rows={4}
            />
            <div className="flex gap-2">
              <Button
                onClick={() => applyMutation.mutate()}
                disabled={!corrections.trim() || applyMutation.isPending}
                size="sm"
              >
                {applyMutation.isPending ? "Applying..." : "Apply corrections"}
              </Button>
              <Button
                onClick={() => setPrompts(null)}
                variant="ghost"
                size="sm"
              >
                Cancel
              </Button>
            </div>
            {applyMutation.isError && (
              <p className="text-destructive text-xs">{(applyMutation.error as Error).message}</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
