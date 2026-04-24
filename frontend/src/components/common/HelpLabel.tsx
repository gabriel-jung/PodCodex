import HelpIcon from "./HelpIcon";

interface HelpLabelProps {
  label: string;
  /** Optional tooltip text. When omitted, the component renders as a plain
   *  row label — same typography, no help icon. */
  help?: string;
}

export default function HelpLabel({ label, help }: HelpLabelProps) {
  return (
    <label className="text-muted-foreground flex items-center gap-1">
      {label}
      {help && <HelpIcon help={help} />}
    </label>
  );
}
