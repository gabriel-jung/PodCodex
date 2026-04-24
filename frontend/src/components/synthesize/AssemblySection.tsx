import type { UseMutationResult } from "@tanstack/react-query";
import type { PipelineConfig, SynthesisStatus } from "@/api/types";
import { Button } from "@/components/ui/button";
import AdvancedToggle from "@/components/common/AdvancedToggle";
import FormGrid from "@/components/common/FormGrid";
import HelpLabel from "@/components/common/HelpLabel";
import SectionHeader from "@/components/common/SectionHeader";
import { errorMessage, selectClass } from "@/lib/utils";

export interface AssemblySectionProps {
  // Settings state
  assembleStrategy: string;
  setAssembleStrategy: (v: string) => void;
  silenceDuration: number;
  setSilenceDuration: (v: number) => void;

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
  silenceDuration,
  setSilenceDuration,
  pipelineConfig,
  status,
  assembleMutation,
}: AssemblySectionProps) {
  return (
    <section className="space-y-3 border-t border-border/50 pt-3">
      <SectionHeader help="Stitch generated segments into the final episode audio.">
        4. Assemble
      </SectionHeader>

      <FormGrid>
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
      </FormGrid>

      <AdvancedToggle className="space-y-3">
        <FormGrid className="pl-3 border-l-2 border-border">
          <HelpLabel label="Silence (s)" help="Pause inserted between segments when the strategy stitches them back-to-back." />
          <input
            type="number"
            value={silenceDuration}
            onChange={(e) => setSilenceDuration(Number(e.target.value))}
            step={0.1}
            min={0}
            max={5}
            className="input w-20"
          />
        </FormGrid>
      </AdvancedToggle>

      <div className="flex items-center gap-3 flex-wrap">
        <Button
          onClick={() => assembleMutation.mutate()}
          disabled={!status?.tts_segments_generated || assembleMutation.isPending}
          size="sm"
        >
          {assembleMutation.isPending ? "Assembling..." : "Assemble"}
        </Button>
        {!status?.tts_segments_generated && (
          <span className="text-xs text-muted-foreground">Generate TTS segments first</span>
        )}
        {assembleMutation.isError && (
          <span className="text-xs text-destructive w-full">
            {errorMessage(assembleMutation.error)}
          </span>
        )}
      </div>
    </section>
  );
}
