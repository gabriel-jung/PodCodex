interface SectionHeaderProps {
  children: React.ReactNode;
  className?: string;
}

/** Consistent section heading used across pipeline panels. */
export default function SectionHeader({ children, className = "" }: SectionHeaderProps) {
  return (
    <h5 className={`text-xs font-medium text-muted-foreground ${className}`}>
      {children}
    </h5>
  );
}
