/** Pipeline setting rows shared by the app defaults (Settings → Pipeline) and
 *  a show's overrides (show Settings).
 *
 *  Each field owns its label, help text and options once. The two pages keep
 *  their own storage: the app page writes the store, the show page a local
 *  draft saved to show.toml. On the show page every field also offers
 *  "inherit", passed as `inherit` (the label, naming the live app default)
 *  with the empty value `""`. */

import { useMemo } from "react";
import { SettingRow } from "@/components/ui/setting-row";
import { ErrorAlert } from "@/components/ui/error-alert";
import { inputWidth, selectClass } from "@/lib/utils";
import { useLLMProviders } from "@/hooks/useLLMProviders";
import { useProviderProfiles } from "@/hooks/useProviderProfiles";
import { useApiKeys } from "@/hooks/useApiKeys";
import { useIndexConfig } from "@/hooks/useIndexConfig";
import { LLM_MODE_LABELS } from "@/lib/pipelineLabels";

/** The "App default (…)" option of an overridable field, or undefined on the
 *  app page itself. */
type Inherit = string | undefined;

interface SelectFieldProps {
  value: string;
  onChange: (v: string) => void;
  inherit?: Inherit;
}

function InheritOption({ inherit }: { inherit: Inherit }) {
  return inherit === undefined ? null : <option value="">{inherit}</option>;
}

/** A setting whose options come from a server read: disabled on
 *  "Loading…" until the read lands, and a failed read shown under the row
 *  with Retry instead of a select stuck loading. */
function LoadedSelectField({
  label,
  help,
  value,
  onChange,
  inherit,
  options,
  error,
  onRetry,
}: SelectFieldProps & {
  label: string;
  help: string;
  /** `[value, label]` pairs, or undefined while the read is in flight. */
  options: [string, string][] | undefined;
  error: unknown;
  onRetry: () => void;
}) {
  return (
    <SettingRow
      label={label}
      help={help}
      below={!options && error ? <ErrorAlert error={error} onRetry={onRetry} compact /> : undefined}
    >
      <select value={value} onChange={(e) => onChange(e.target.value)} className={selectClass} disabled={!options}>
        <InheritOption inherit={inherit} />
        {options
          ? options.map(([key, text]) => <option key={key} value={key}>{text}</option>)
          : <option>Loading…</option>}
      </select>
    </SettingRow>
  );
}

export function TranscriptionModelField(props: SelectFieldProps) {
  const { whisperModels, pipelineConfig, error, refetch } = useLLMProviders();
  return (
    <LoadedSelectField
      {...props}
      label="Transcription model"
      help="Bigger models are more accurate but slower. The CPU options work without a graphics card; the GPU options need one."
      options={pipelineConfig ? Object.entries(whisperModels) : undefined}
      error={error}
      onRetry={() => void refetch()}
    />
  );
}

/** An On/Off setting; `value` is "yes", "no", or "" for inherit. */
export function OnOffField({
  label,
  help,
  below,
  value,
  onChange,
  inherit,
}: SelectFieldProps & { label: string; help: string; below?: React.ReactNode }) {
  return (
    <SettingRow label={label} help={help} below={below}>
      <select value={value} onChange={(e) => onChange(e.target.value)} className={selectClass}>
        <InheritOption inherit={inherit} />
        <option value="no">Off</option>
        <option value="yes">On</option>
      </select>
    </SettingRow>
  );
}

export function IdentifySpeakersField(props: SelectFieldProps & { below?: React.ReactNode }) {
  return (
    <OnOffField
      label="Identify speakers"
      help="Detect who is talking and label each line. Requires a Hugging Face token."
      {...props}
    />
  );
}

export function LlmModeField({ value, onChange, inherit }: SelectFieldProps) {
  return (
    <SettingRow
      label="Where the AI runs"
      help="Ollama runs on your own computer. Cloud API uses a paid online provider. Manual lets you copy the prompts and run them yourself."
    >
      <select value={value} onChange={(e) => onChange(e.target.value)} className={selectClass}>
        <InheritOption inherit={inherit} />
        {Object.entries(LLM_MODE_LABELS).map(([mode, label]) => (
          <option key={mode} value={mode}>{label}</option>
        ))}
      </select>
    </SettingRow>
  );
}

export function ProviderField({ value, onChange, inherit }: SelectFieldProps) {
  const { profiles, error } = useProviderProfiles();
  const apiProfiles = useMemo(() => profiles.filter((p) => p.type !== "ollama"), [profiles]);
  return (
    <SettingRow label="AI provider" help="Which provider profile to use. Manage profiles in Settings → Credentials.">
      <select value={value} onChange={(e) => onChange(e.target.value)} className={selectClass}>
        {inherit !== undefined ? <InheritOption inherit={inherit} /> : (
          <option value="">{error ? "Could not load profiles" : "Pick…"}</option>
        )}
        {apiProfiles.map((p) => (
          <option key={p.name} value={p.name}>
            {p.name}{p.builtin ? "" : " (custom)"}
          </option>
        ))}
      </select>
    </SettingRow>
  );
}

export function ApiKeyField({ value, onChange, inherit }: SelectFieldProps) {
  const { keys, error } = useApiKeys();
  return (
    <SettingRow label="AI API key" help="Which saved API key to use. Add keys in Settings → Credentials.">
      <select value={value} onChange={(e) => onChange(e.target.value)} className={selectClass}>
        {inherit !== undefined ? <InheritOption inherit={inherit} /> : (
          <option value="">{error ? "Could not load keys" : keys.length === 0 ? "No keys yet" : "Pick…"}</option>
        )}
        {keys.map((k) => (
          <option key={k.name} value={k.name}>
            {k.name}
            {k.suggested_provider ? ` (${k.suggested_provider})` : ""}
          </option>
        ))}
      </select>
    </SettingRow>
  );
}

/** Text and number fields show an inherited value as their placeholder, the
 *  bare value (the inputs are too narrow for "App default (…)", which goes in
 *  the tooltip). */
interface InputFieldProps {
  value: string;
  onChange: (v: string) => void;
  /** The app default this show inherits when the field is empty. */
  inheritValue?: string;
}

const inheritTitle = (v: string | undefined) => (v === undefined ? undefined : `Empty: app default${v ? ` (${v})` : ""}`);

export function LlmModelField({
  mode,
  value,
  onChange,
  inheritValue,
}: InputFieldProps & { mode: "api" | "ollama" }) {
  const example = mode === "ollama" ? "e.g. llama3.1:8b" : "e.g. gpt-4o-mini";
  return (
    <SettingRow
      label="AI model"
      help={mode === "ollama"
        ? "Model tag served by your local Ollama instance."
        : "Specific model name. Leave blank to use the provider's default."}
    >
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={inheritValue === undefined ? example : inheritValue || "Provider default"}
        title={inheritTitle(inheritValue)}
        className={`input ${inputWidth.medium}`}
      />
    </SettingRow>
  );
}

export function TargetLangField({ value, onChange, inheritValue }: InputFieldProps) {
  return (
    <SettingRow label="Translate into" help="Language episodes are translated into.">
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={inheritValue ?? "English"}
        title={inheritTitle(inheritValue)}
        className={`input ${inputWidth.short}`}
      />
    </SettingRow>
  );
}

/** Minutes per LLM batch. `value` is the raw input text; a page commits it
 *  when it parses to a positive number (or, on the show page, when empty). */
export function BatchMinutesField({ value, onChange, inheritValue }: InputFieldProps) {
  return (
    <SettingRow
      label="Minutes per batch"
      help="Max audio duration per LLM request. Smaller batches stay within model context windows; larger batches are fewer requests but heavier prompts. An episode panel can override it per run as a batch count."
    >
      <input
        type="number"
        min={1}
        step={1}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={inheritValue}
        title={inheritTitle(inheritValue)}
        className={`input ${inputWidth.numeric}`}
      />
    </SettingRow>
  );
}

export function EmbeddingModelField(props: SelectFieldProps) {
  const { data, error, refetch } = useIndexConfig();
  return (
    <LoadedSelectField
      {...props}
      label="Embedding model"
      help="The model that turns transcript text into vectors so search can find passages by meaning. AI search queries whichever model is set here."
      options={data ? Object.entries(data.models).map(([key, m]) => [key, m.label]) : undefined}
      error={error}
      onRetry={() => void refetch()}
    />
  );
}

export function ChunkingField(props: SelectFieldProps) {
  const { data, error, refetch } = useIndexConfig();
  return (
    <LoadedSelectField
      {...props}
      label="Chunking"
      help="How transcripts are split before they are embedded. Semantic groups sentences with similar meaning; speaker groups consecutive lines from the same speaker."
      options={data ? Object.keys(data.chunking_strategies).map((key) => [key, key]) : undefined}
      error={error}
      onRetry={() => void refetch()}
    />
  );
}
