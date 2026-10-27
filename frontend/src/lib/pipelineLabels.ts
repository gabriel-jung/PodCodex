/** Display labels for pipeline setting values, shared by the settings forms
 *  (the option text) and the show page's "App default (…)" choices, so a
 *  renamed option reads the same in both. */

export const LLM_MODE_LABELS = {
  api: "Cloud API",
  ollama: "Ollama (local)",
  manual: "Manual (copy-paste prompts)",
} as const;
