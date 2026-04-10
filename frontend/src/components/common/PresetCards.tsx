/**
 * Reusable preset card grid — shared between the batch StepConfigEditor and
 * the per-episode pipeline panels (Transcribe / Correct / Translate / Index)
 * so both surfaces expose the same preset shortcuts.
 */

export default function PresetCards<K extends string>({
  presets,
  active,
  onSelect,
  className = "",
}: {
  presets: Record<K, { label: string; desc: string }>;
  active: string;
  onSelect: (key: K) => void;
  className?: string;
}) {
  const count = Object.keys(presets).length;
  return (
    <div className={`grid gap-2 items-stretch ${count === 2 ? "grid-cols-2" : "grid-cols-3"} ${className}`}>
      {(Object.entries(presets) as [K, { label: string; desc: string }][]).map(([key, p]) => (
        <button
          key={key}
          onClick={() => onSelect(key)}
          className={`rounded-lg border p-3 text-left transition flex flex-col ${
            active === key
              ? "border-primary bg-accent"
              : "border-border hover:border-primary/50"
          }`}
        >
          <div className="text-sm font-medium">{p.label}</div>
          <div className="text-xs text-muted-foreground mt-0.5">{p.desc}</div>
        </button>
      ))}
    </div>
  );
}
