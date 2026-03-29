import type { UseMutationResult } from "@tanstack/react-query";
import type { PipelineConfig, SynthesisStatus } from "@/api/types";
import { Button } from "@/components/ui/button";
import SectionHeader from "@/components/common/SectionHeader";
import HelpLabel from "@/components/common/HelpLabel";
import { errorMessage, selectClass } from "@/lib/utils";

export interface AssemblySectionProps {
  // Settings state
  assembleStrategy: string;
  setAssembleStrategy: (v: string) => void;

  // Pipeline config
  pipelineConfig: PipelineConfig | undefined;

  // Status
  status: SynthesisStatus | undefined;

  // Mutation
  assembleMutation: UseMutationResult<{ path: string; duration: number }, Error, void>;
}

export default function AssemblySection({
  assembleStrategy,
  setAssembleStrategy,
  pipelineConfig,
  status,
  assembleMutation,
}: AssemblySectionProps) {
  return (
    <section className="space-y-3 border-t border-border/50 pt-3">
      <SectionHeader>3. Assemble Episode</SectionHeader>

      <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-x-4 gap-y-2 sm:gap-y-3 items-start sm:items-center text-sm">
        <HelpLabel label="Strategy" help="How to handle pauses between segments in the final audio." />
        <select
          value={assembleStrategy}
          onChange={(e) => setAssembleStrategy(e.target.value)}
          className={selectClass}
        >
          {pipelineConfig
            ? Object.entries(pipelineConfig.assemble_strategies).map(([key, desc]) => (
                <option key={key} value={key} title={desc}>{desc}</option>
              ))
            : <option value={assembleStrategy}>{assembleStrategy}</option>
          }
        </select>
      </div>

      <div className="flex items-center gap-3">
        <Button
          onClick={() => assembleMutation.mutate()}
          disabled={!status?.tts_segments_generated || assembleMutation.isPending}
          size="sm"
        >
          {assembleMutation.isPending ? "Assembling..." : "Assemble"}
        </Button>
        {assembleMutation.isError && (
          <span className="text-xs text-destructive">
            {errorMessage(assembleMutation.error)}
          </span>
        )}
      </div>
    </section>
  );
}
