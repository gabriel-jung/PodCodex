import type { UseMutationResult } from "@tanstack/react-query";
import type { PipelineConfig, GeneratedSegment, SynthesisStatus } from "@/api/types";
import { Button } from "@/components/ui/button";
import SectionHeader from "@/components/common/SectionHeader";
import HelpLabel from "@/components/common/HelpLabel";
import AdvancedToggle from "@/components/common/AdvancedToggle";
import { errorMessage, selectClass } from "@/lib/utils";

export interface TTSGenerationSectionProps {
  // Settings state
  language: string;
  setLanguage: (v: string) => void;
  modelSize: string;
  setModelSize: (v: string) => void;
  sourceLang: string;
  setSourceLang: (v: string) => void;
  maxChunkDuration: number;
  setMaxChunkDuration: (v: number) => void;

  // Episode data
  translations: string[];

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
  sourceLang,
  setSourceLang,
  maxChunkDuration,
  setMaxChunkDuration,
  translations,
  pipelineConfig,
  status,
  generatedSegments,
  generateMutation,
}: TTSGenerationSectionProps) {
  return (
    <section className="space-y-3 border-t border-border/50 pt-3">
      <SectionHeader>2. Generate TTS Segments</SectionHeader>

      <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-x-4 gap-y-2 sm:gap-y-3 items-start sm:items-center text-sm">
        <HelpLabel label="Language" help="The language the generated speech should sound like." />
        <input
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          className="input py-1 text-sm"
        />
        <HelpLabel label="Model size" help="Larger model produces more natural-sounding speech but needs more GPU memory." />
        <select
          value={modelSize}
          onChange={(e) => setModelSize(e.target.value)}
          className={selectClass}
        >
          {pipelineConfig
            ? Object.entries(pipelineConfig.tts_model_sizes).map(([key, desc]) => (
                <option key={key} value={key}>{key} — {desc}</option>
              ))
            : <option value={modelSize}>{modelSize}</option>
          }
        </select>
        {translations.length > 0 && (
          <>
            <HelpLabel label="Text source" help="Which text to synthesize. 'Best available' picks the translation if one exists, otherwise the polished or raw transcript." />
            <select
              value={sourceLang}
              onChange={(e) => setSourceLang(e.target.value)}
              className={selectClass}
            >
              <option value="">Best available</option>
              {translations.map((lang) => (
                <option key={lang} value={lang}>{lang}</option>
              ))}
            </select>
          </>
        )}
      </div>

      {/* Advanced TTS settings */}
      <AdvancedToggle className="space-y-3">
        <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-x-4 gap-y-2 sm:gap-y-3 items-start sm:items-center text-sm pl-3 border-l-2 border-border">
          <HelpLabel label="Max chunk (s)" help="Maximum duration in seconds for each TTS chunk. Shorter chunks are more stable, longer ones sound more natural." />
          <input
            type="number"
            value={maxChunkDuration}
            onChange={(e) => setMaxChunkDuration(Number(e.target.value))}
            className="input py-1 text-sm w-20"
            min={5}
            max={60}
          />
        </div>
      </AdvancedToggle>

      <div className="flex items-center gap-3">
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
        {status?.tts_segments_generated && (
          <span className="text-xs text-green-400">
            {generatedSegments?.length ?? "?"} segments generated
          </span>
        )}
        {generateMutation.isError && (
          <span className="text-xs text-destructive">
            {errorMessage(generateMutation.error)}
          </span>
        )}
      </div>
    </section>
  );
}
