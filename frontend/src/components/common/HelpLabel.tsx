import HelpIcon from "./HelpIcon";

interface HelpLabelProps {
  label: string;
  /** Optional tooltip text. When omitted, the component renders as a plain
   *  row label — same typography, no help icon. */
  help?: string;
  /** Id of the control this labels. The label then names that control and
   *  clicking it focuses the control. */
  htmlFor?: string;
}

/** Row label with an optional (?) help icon. The icon sits beside the
 *  `<label>`, not inside it: a label names its first labelable descendant,
 *  so wrapping the icon's button made every field's label name the button. */
export default function HelpLabel({ label, help, htmlFor }: HelpLabelProps) {
  return (
    <span className="text-xs text-muted-foreground flex items-center gap-1">
      <label htmlFor={htmlFor}>{label}</label>
      {help && <HelpIcon help={help} />}
    </span>
  );
}
