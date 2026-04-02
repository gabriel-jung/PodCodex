/** Consistent two-column form grid used across pipeline panels: label on the left, control on the right. */

interface FormGridProps {
  children: React.ReactNode;
  /** Additional classes (e.g. "max-w-lg", "pl-3 border-l-2 border-border"). */
  className?: string;
}

export default function FormGrid({ children, className }: FormGridProps) {
  return (
    <div className={`grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-x-4 gap-y-2 sm:gap-y-3 items-start sm:items-center text-sm ${className ?? ""}`}>
      {children}
    </div>
  );
}
