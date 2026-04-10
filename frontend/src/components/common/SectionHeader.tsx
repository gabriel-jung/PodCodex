interface SectionHeaderProps {
  children: React.ReactNode;
}

/** Consistent section heading used across pipeline panels. */
export default function SectionHeader({ children }: SectionHeaderProps) {
  return (
    <h5 className="text-xs font-medium text-muted-foreground/70">
      {children}
    </h5>
  );
}
