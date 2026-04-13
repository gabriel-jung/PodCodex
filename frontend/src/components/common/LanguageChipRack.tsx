// Labels are stored verbatim into LLMConfig.sourceLang / pipelineConfigStore.targetLang
// and interpolated into LLM prompt templates — keep them English (what the prompt
// expects) rather than native names like `Français`.
export const TOP_LANGS: readonly string[] = ["English", "French", "German", "Spanish", "Italian"];

interface LanguageChipRackProps {
  value: string;
  onChange: (value: string) => void;
  /** Languages to show as chips. Defaults to TOP_LANGS. */
  options?: readonly string[];
  otherPlaceholder?: string;
}

/**
 * Chip rack for picking a language: 5 top chips + an "Other" chip that reveals
 * a free-text input. "Other" mode is derived from the value — anything not in
 * the chip list counts as Other (including the empty string that Other-click sets).
 */
export default function LanguageChipRack({
  value,
  onChange,
  options = TOP_LANGS,
  otherPlaceholder = "e.g. Japanese, Arabic…",
}: LanguageChipRackProps) {
  const isOther = !options.includes(value);
  return (
    <div className="flex flex-wrap gap-1.5">
      {options.map((label) => {
        const selected = !isOther && value === label;
        return (
          <button
            key={label}
            type="button"
            onClick={() => onChange(label)}
            className={`px-2.5 py-1 text-xs rounded-md border transition ${selected ? "bg-primary text-primary-foreground border-primary" : "border-border hover:bg-accent"}`}
          >
            {label}
          </button>
        );
      })}
      <button
        type="button"
        onClick={() => onChange("")}
        className={`px-2.5 py-1 text-xs rounded-md border transition ${isOther ? "bg-primary text-primary-foreground border-primary" : "border-border hover:bg-accent"}`}
      >
        Other
      </button>
      {isOther && (
        <input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={otherPlaceholder}
          className="input text-xs w-40"
          autoFocus
        />
      )}
    </div>
  );
}
