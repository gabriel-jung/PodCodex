/** Segmented toggle used across pipeline panel forms (Source, Mode, Cleanup, …).
 *
 *  Each option is a tuple: `[key, label, title?, enabled?]`. A disabled option
 *  stays visible but renders muted and can't be clicked — useful when a mode
 *  is currently unavailable (e.g. "Audio" when no audio file is present).
 */
export default function Segmented<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (v: T) => void;
  options: readonly (readonly [T, string, string?, boolean?])[];
}) {
  return (
    <div className="inline-flex rounded-md border border-border overflow-hidden text-xs w-fit">
      {options.map(([key, label, title, enabled = true]) => (
        <button
          key={key}
          onClick={() => enabled && onChange(key)}
          disabled={!enabled}
          title={title}
          className={`px-3 py-1 transition ${
            value === key
              ? "bg-accent font-medium"
              : enabled
                ? "hover:bg-accent/50 text-muted-foreground"
                : "text-muted-foreground/40 cursor-not-allowed"
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
