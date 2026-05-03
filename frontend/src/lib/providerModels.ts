/**
 * Popular models per provider, surfaced as `<datalist>` autocomplete hints
 * in the LLM model input. The input stays free-text so any model the
 * provider supports works without a code change. Update lazily.
 */
// First entry is the canonical default — kept in sync with
// ``LLM_PROVIDER_DEFAULTS`` in ``src/podcodex/core/constants.py`` so the
// autocomplete top-suggestion matches the runtime fallback when the user
// leaves the model field blank.
const PROVIDER_MODELS: Record<string, readonly string[]> = {
  openai: ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "o4-mini"],
  anthropic: ["claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5"],
  mistral: ["mistral-small-latest", "mistral-large-latest", "codestral-latest"],
  deepseek: ["deepseek-chat", "deepseek-reasoner"],
  gemini: ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
  groq: [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen-2.5-72b-instruct",
    "deepseek-r1-distill-llama-70b",
  ],
  openrouter: [
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-4o",
    "deepseek/deepseek-chat",
    "google/gemini-2.5-flash",
  ],
};

/** Suggested models for the named provider profile. Empty array if unknown. */
export function modelsFor(profileName: string | null | undefined): readonly string[] {
  if (!profileName) return [];
  return PROVIDER_MODELS[profileName] ?? [];
}

/** Provider-aware placeholder for the model input. */
export function modelPlaceholderFor(profileName: string | null | undefined): string {
  const models = modelsFor(profileName);
  if (models.length > 0) return `e.g. ${models[0]}`;
  return "e.g. gpt-4o-mini";
}
