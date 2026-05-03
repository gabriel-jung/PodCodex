import { useEffect, useState } from "react";

interface NumberInputProps {
  value: number;
  onChange: (n: number) => void;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
  disabled?: boolean;
  placeholder?: string;
  "aria-label"?: string;
}

/** Number input that lets users clear and retype freely. Commits valid values
 *  as the user types, clamps to [min, max] on blur. */
export function NumberInput({
  value,
  onChange,
  min,
  max,
  step,
  className,
  disabled,
  placeholder,
  "aria-label": ariaLabel,
}: NumberInputProps) {
  const [draft, setDraft] = useState(String(value));

  useEffect(() => {
    setDraft(String(value));
  }, [value]);

  const inRange = (n: number) =>
    (min == null || n >= min) && (max == null || n <= max);

  return (
    <input
      type="number"
      value={draft}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      placeholder={placeholder}
      aria-label={ariaLabel}
      className={className}
      onChange={(e) => {
        const v = e.target.value;
        setDraft(v);
        if (v === "") return;
        const n = Number(v);
        if (Number.isFinite(n) && inRange(n)) onChange(n);
      }}
      onBlur={() => {
        const n = Number(draft);
        if (!Number.isFinite(n) || draft === "") {
          const fallback = min ?? 0;
          onChange(fallback);
          setDraft(String(fallback));
          return;
        }
        if (min != null && n < min) {
          onChange(min);
          setDraft(String(min));
        } else if (max != null && n > max) {
          onChange(max);
          setDraft(String(max));
        } else {
          setDraft(String(n));
        }
      }}
    />
  );
}

interface NullableNumberInputProps extends Omit<NumberInputProps, "value" | "onChange"> {
  /** ``null`` means "no value" — empty input, placeholder shown. */
  value: number | null;
  onChange: (n: number | null) => void;
}

/** Same as ``NumberInput`` but treats an empty input as ``null`` rather than
 *  falling back to ``min``. Useful for "Auto"-style fields where blank means
 *  "let the backend decide." */
export function NullableNumberInput({
  value,
  onChange,
  min,
  max,
  step,
  className,
  disabled,
  placeholder,
  "aria-label": ariaLabel,
}: NullableNumberInputProps) {
  const [draft, setDraft] = useState(value == null ? "" : String(value));

  useEffect(() => {
    setDraft(value == null ? "" : String(value));
  }, [value]);

  const inRange = (n: number) =>
    (min == null || n >= min) && (max == null || n <= max);

  return (
    <input
      type="number"
      value={draft}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      placeholder={placeholder}
      aria-label={ariaLabel}
      className={className}
      onChange={(e) => {
        const v = e.target.value;
        setDraft(v);
        if (v === "") {
          onChange(null);
          return;
        }
        const n = Number(v);
        if (Number.isFinite(n) && inRange(n)) onChange(n);
      }}
      onBlur={() => {
        if (draft === "") {
          onChange(null);
          return;
        }
        const n = Number(draft);
        if (!Number.isFinite(n)) {
          onChange(null);
          setDraft("");
          return;
        }
        if (min != null && n < min) {
          onChange(min);
          setDraft(String(min));
        } else if (max != null && n > max) {
          onChange(max);
          setDraft(String(max));
        } else {
          setDraft(String(n));
        }
      }}
    />
  );
}
