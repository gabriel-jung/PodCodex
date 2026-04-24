import HelpIcon from "./HelpIcon";

interface SectionHeaderProps {
  children: React.ReactNode;
  /** Optional help text shown via a (?) icon + tooltip next to the title. */
  help?: string;
  className?: string;
}

/** Consistent section heading used across pipeline panels. */
export default function SectionHeader({ children, help, className = "" }: SectionHeaderProps) {
  return (
    <h5 className={`text-xs font-medium text-muted-foreground flex items-center gap-1 ${className}`}>
      <span>{children}</span>
      {help && <HelpIcon help={help} />}
    </h5>
  );
}
