/** Inline notice shown in a pipeline panel when the working run config for
 *  this step differs from the show's saved config. Lets the user push the
 *  changed values into show.toml ("Save to show") or revert them ("Reset").
 *  A panel edit is otherwise a per-run tweak that never reaches the show. */

import { useMemo } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUpToLine, Undo2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { getShowMeta, updateShowMeta } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { useEpisodeStore } from "@/stores";
import {
  usePipelineConfigStore,
  effectiveBundle,
  type ConfigBundle,
} from "@/stores/pipelineConfigStore";
import type { PipelineInputStep } from "@/lib/pipelineInputs";
import type { PipelineDefaults } from "@/api/types";

/** The show.toml [pipeline] fields a given step owns. */
function stepFields(step: PipelineInputStep, b: ConfigBundle): Partial<PipelineDefaults> {
  if (step === "transcribe") {
    return {
      model_size: b.transcribe.modelSize,
      diarize: b.transcribe.diarize,
      num_speakers: b.transcribe.numSpeakers,
    };
  }
  if (step === "index") return { rag_model: b.indexModel, rag_chunker: b.indexChunker };
  // `context` is a per-episode merge of show description + episode title and
  // description. It would never round-trip cleanly to show.toml, so the
  // banner ignores it; show-level context lives in ShowSettings.
  const llm: Partial<PipelineDefaults> = {
    llm_mode: b.llm.mode,
    llm_provider_profile: b.llm.providerProfile,
    llm_key_name: b.llm.keyName,
    // Only the active mode's model is part of this step's run-config diff;
    // other modes' stashes are unrelated to what's about to run.
    llm_models_by_mode: { [b.llm.mode]: b.llm.model },
    llm_batch_minutes: b.llm.batchMinutes,
  };
  return step === "translate" ? { ...llm, target_lang: b.targetLang } : llm;
}

export default function RunSettingsBanner({ step }: { step: PipelineInputStep }) {
  const queryClient = useQueryClient();
  const folder = useEpisodeStore((s) => s.folder);
  const appDefaults = usePipelineConfigStore((s) => s.appDefaults);
  const seedWorkingFromShow = usePipelineConfigStore((s) => s.seedWorkingFromShow);
  const transcribe = usePipelineConfigStore((s) => s.transcribe);
  const llm = usePipelineConfigStore((s) => s.llm);
  const targetLang = usePipelineConfigStore((s) => s.targetLang);
  const indexModel = usePipelineConfigStore((s) => s.indexModel);
  const indexChunker = usePipelineConfigStore((s) => s.indexChunker);

  const { data: meta } = useQuery({
    queryKey: queryKeys.showMeta(folder ?? ""),
    queryFn: () => getShowMeta(folder!),
    enabled: !!folder,
  });

  // appDefaults supplies fields stepFields() does not read (engine, presets);
  // the working slices override the rest.
  const workingBundle = useMemo<ConfigBundle>(() => ({
    ...appDefaults,
    transcribe,
    llm,
    targetLang,
    indexModel,
    indexChunker,
  }), [appDefaults, transcribe, llm, targetLang, indexModel, indexChunker]);

  const save = useMutation({
    mutationFn: () => {
      const patch = stepFields(step, workingBundle);
      // Per-mode model dict must merge with existing entries rather than
      // clobber them; the patch only carries the active mode.
      const existingModels = meta!.pipeline?.llm_models_by_mode ?? {};
      const incomingModels = patch.llm_models_by_mode ?? {};
      const mergedModels = { ...existingModels, ...incomingModels };
      return updateShowMeta(folder!, {
        ...meta!,
        pipeline: {
          ...meta!.pipeline,
          ...patch,
          llm_models_by_mode: mergedModels,
        },
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.showMeta(folder ?? "") });
      queryClient.invalidateQueries({ queryKey: queryKeys.shows() });
    },
  });

  if (!folder || !meta) return null;

  const effective = effectiveBundle(appDefaults, meta.pipeline);
  const workingFields = JSON.stringify(stepFields(step, workingBundle));
  const effectiveFields = JSON.stringify(stepFields(step, effective));
  if (workingFields === effectiveFields) return null;

  return (
    <div className="flex items-center gap-2 rounded border border-info/30 bg-info/10 px-3 py-1.5 text-xs">
      <span className="flex-1 text-muted-foreground">
        Run settings changed. They apply to this run only unless you save them
        to the show.
      </span>
      <Button
        onClick={() => save.mutate()}
        disabled={save.isPending}
        variant="ghost"
        size="sm"
        className="h-6 px-2 text-xs gap-1 shrink-0"
      >
        <ArrowUpToLine className="w-3 h-3" />
        {save.isPending ? "Saving…" : "Save to show"}
      </Button>
      <Button
        onClick={() => seedWorkingFromShow(meta.pipeline)}
        variant="ghost"
        size="sm"
        className="h-6 px-2 text-xs gap-1 shrink-0"
      >
        <Undo2 className="w-3 h-3" /> Reset
      </Button>
    </div>
  );
}
