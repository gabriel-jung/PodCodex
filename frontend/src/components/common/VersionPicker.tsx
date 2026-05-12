/**
 * Single facility for "pick a version" dropdowns across the app
 * (transcript / correct / translate / synthesize sources, synth output
 * versions, editor toolbar, batch per-episode pickers, diff reference
 * pickers). One implementation of:
 *   - "Latest, <first version label>" header + slice(1) (the empty-value
 *     option means "track latest"; never duplicated as a real option),
 *   - optional prepended sentinel options ("None", "Original", …) for
 *     reference pickers,
 *   - optional width-fit trigger so the dropdown only takes the space of
 *     the currently-selected label (toolbar contexts).
 */

import type { VersionEntry } from "@/api/types";
import { selectClass, versionOption } from "@/lib/utils";
import AutoWidthSelect from "./AutoWidthSelect";

export interface PrependOption {
  value: string;
  label: string;
}

export interface VersionPickerProps {
  versions: VersionEntry[];
  value: string | null;
  onChange: (v: string | null) => void;
  /** Show the "Latest, <first version>" header bound to value="". Default true. */
  showLatest?: boolean;
  /** Sentinel options rendered before the version list (e.g. None, Original). */
  prependOptions?: PrependOption[];
  /** Wrap in AutoWidthSelect so the closed trigger fits its selected label. */
  widthFit?: boolean;
  /** Override the label used for the closed trigger; only consumed when widthFit. */
  selectedLabel?: string;
  /** Message when there is nothing to pick (no versions AND no prepend options). */
  emptyMessage?: string;
  title?: string;
  /** Applied to the bare <select>; ignored when widthFit (AutoWidthSelect owns layout). */
  className?: string;
}

function deriveTriggerLabel(
  value: string | null,
  versions: VersionEntry[],
  prependOptions: PrependOption[] | undefined,
  showLatest: boolean,
): string {
  if (value == null) {
    if (showLatest && versions.length > 0) return `Latest, ${versionOption(versions[0])}`;
    return prependOptions?.[0]?.label ?? "";
  }
  const fromPrepend = prependOptions?.find((p) => p.value === value);
  if (fromPrepend) return fromPrepend.label;
  const v = versions.find((vv) => vv.id === value);
  return v ? versionOption(v) : "";
}

export default function VersionPicker({
  versions,
  value,
  onChange,
  showLatest = true,
  prependOptions,
  widthFit = false,
  selectedLabel,
  emptyMessage = "No versions available yet.",
  title,
  className,
}: VersionPickerProps) {
  const hasPrepend = !!prependOptions && prependOptions.length > 0;
  if (versions.length === 0 && !hasPrepend) {
    return <p className="text-xs text-muted-foreground italic">{emptyMessage}</p>;
  }

  const versionList = showLatest ? versions.slice(1) : versions;
  const stringValue = value ?? "";

  const options = (
    <>
      {showLatest && versions.length > 0 && (
        <option value="">Latest, {versionOption(versions[0])}</option>
      )}
      {prependOptions?.map((p) => (
        <option key={`__prepend_${p.value}`} value={p.value}>
          {p.label}
        </option>
      ))}
      {versionList.map((v) => (
        <option key={v.id} value={v.id}>
          {versionOption(v)}
        </option>
      ))}
    </>
  );

  if (widthFit) {
    const triggerLabel =
      selectedLabel ?? deriveTriggerLabel(value, versions, prependOptions, showLatest);
    return (
      <AutoWidthSelect
        value={stringValue}
        onChange={(v) => onChange(v || null)}
        selectedLabel={triggerLabel}
        title={title}
      >
        {options}
      </AutoWidthSelect>
    );
  }

  return (
    <select
      value={stringValue}
      onChange={(e) => onChange(e.target.value || null)}
      title={title}
      className={className ?? `${selectClass} text-xs max-w-full min-w-0`}
    >
      {options}
    </select>
  );
}
