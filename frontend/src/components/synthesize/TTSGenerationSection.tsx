import type { UseMutationResult } from "@tanstack/react-query";
import type { PipelineConfig, GeneratedSegment, SynthesisStatus } from "@/api/types";
import { Button } from "@/components/ui/button";
import AdvancedToggle from "@/components/common/AdvancedToggle";
import FormGrid from "@/components/common/FormGrid";
import HelpLabel from "@/components/common/HelpLabel";
import LanguageChipRack from "@/components/common/LanguageChipRack";
import SectionHeader from "@/components/common/SectionHeader";
import { errorMessage, selectClass } from "@/lib/utils";

export interface TTSGenerationSectionProps {
  // Settings state
  language: string;
  setLanguage: (v: string) => void;
  modelSize: string;
  setModelSize: (v: string) => void;
  maxChunkDuration: number;
  setMaxChunkDuration: (v: number) => void;
  force: boolean;
  setForce: (v: boolean) => void;
  onlySpeakers: string[];
  setOnlySpeakers: (v: string[]) => void;

  // Episode data — speakers for the "only" filter
  allSpeakers: string[];

  // Source label for the generate footer (resolved upstream).
  sourceSummary: string;

  // Pipeline config
  pipelineConfig: PipelineConfig | undefined;

  // Status & generated data
  status: SynthesisStatus | undefined;
  generatedSegments: GeneratedSegment[] | undefined;

  // Mutation
  generateMutation: UseMutationResult<unknown, Error, void>;
}

export default function TTSGenerationSection({
  language,
  setLanguage,
  modelSize,
  setModelSize,
  maxChunkDuration,
  setMaxChunkDuration,
  force,
  setForce,
  onlySpeakers,
  setOnlySpeakers,
  allSpeakers,
  sourceSummary,
  pipelineConfig,
  status,
  generatedSegments,
  generateMutation,
}: TTSGenerationSectionProps) {
  const toggleSpeaker = (sp: string) => {
    setOnlySpeakers(
      onlySpeakers.includes(sp)
        ? onlySpeakers.filter((s) => s !== sp)
        : [...onlySpeakers, sp],
    );
  };

  return (
    <section className="space-y-3 border-t border-border/50 pt-3">
      <SectionHeader help="Render TTS audio for each kept segment, using the cloned voices.">
        3. Generate
      </SectionHeader>

      <FormGrid>
        <HelpLabel label="Voice language" help="The language the cloned voices should speak in. Should match the source text." />
        <LanguageChipRack value={language} onChange={setLanguage} />

        <HelpLabel label="Model size" help="Larger models produce more natural speech but need more GPU memory." />
        <select
          value={modelSize}
          onChange={(e) => setModelSize(e.target.value)}
          className={selectClass}
        >
          {pipelineConfig
            ? Object.entries(pipelineConfig.tts_model_sizes).map(([key, desc]) => (
                <option key={key} value={key}>{key}, {desc}</option>
              ))
            : <option value={modelSize}>{modelSize}</option>
          }
        </select>
      </FormGrid>

      <AdvancedToggle className="space-y-3">
        <FormGrid className="pl-3 border-l-2 border-border">
          <HelpLabel label="Max chunk (s)" help="Maximum duration per TTS chunk in seconds. Shorter chunks are more stable, longer ones sound more natural." />
          <input
            type="number"
            value={maxChunkDuration}
            onChange={(e) => setMaxChunkDuration(Number(e.target.value))}
            className="input w-20"
            min={5}
            max={60}
          />

          {allSpeakers.length > 1 && (
            <>
              <HelpLabel label="Only speakers" help="Restrict generation to selected speakers. Empty = all." />
              <div className="flex flex-wrap gap-1.5">
                {allSpeakers.map((sp) => {
                  const on = onlySpeakers.includes(sp);
                  return (
                    <button
                      key={sp}
                      type="button"
                      onClick={() => toggleSpeaker(sp)}
                      className={`px-2 py-0.5 text-xs rounded-md border transition ${
                        on
                          ? "bg-primary text-primary-foreground border-primary"
                          : "border-border hover:bg-accent"
                      }`}
                    >
                      {sp}
                    </button>
                  );
                })}
                {onlySpeakers.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setOnlySpeakers([])}
                    className="text-xs text-muted-foreground hover:text-foreground px-2"
                  >
                    Clear
                  </button>
                )}
              </div>
            </>
          )}

          <HelpLabel label="Regenerate" help="Re-run every segment even if an identical one already exists on disk." />
          <label className="flex items-center gap-1.5 cursor-pointer text-xs text-muted-foreground w-fit">
            <input
              type="checkbox"
              checked={force}
              onChange={(e) => setForce(e.target.checked)}
              className="accent-primary"
            />
            Force regenerate all
          </label>
        </FormGrid>
      </AdvancedToggle>

      <div className="flex items-center gap-3 flex-wrap">
        <Button
          onClick={() => generateMutation.mutate()}
          disabled={!status?.voice_samples_extracted || generateMutation.isPending}
          size="sm"
        >
          {status?.tts_segments_generated ? "Re-generate" : "Generate"}
        </Button>
        {!status?.voice_samples_extracted && (
          <span className="text-xs text-muted-foreground">Extract voices first</span>
        )}
        {status?.voice_samples_extracted && (
          <span className="text-xs text-muted-foreground">
            From {sourceSummary}
            {onlySpeakers.length > 0 && ` · ${onlySpeakers.length} speaker${onlySpeakers.length !== 1 ? "s" : ""}`}
            {force && " · force"}
          </span>
        )}
        {status?.tts_segments_generated && (
          <span className="text-xs text-success">
            {generatedSegments?.length ?? "?"} segments generated
          </span>
        )}
        {generateMutation.isError && (
          <span className="text-xs text-destructive w-full">
            {errorMessage(generateMutation.error)}
          </span>
        )}
      </div>
    </section>
  );
}
