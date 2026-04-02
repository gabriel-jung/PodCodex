interface SectionHeaderProps {
  children: React.ReactNode;
}

/** Consistent section heading used across pipeline panels. */
export default function SectionHeader({ children }: SectionHeaderProps) {
  return (
    <h5 className="text-[11px] font-medium text-muted-foreground/60 uppercase tracking-wide">
      {children}
    </h5>
  );
}
