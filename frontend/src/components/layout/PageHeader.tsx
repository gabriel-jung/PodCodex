import BackNav from "./BackNav";

interface PageHeaderProps {
  title: string;
  /** BackNav parent breadcrumb. Omit to hide the back button. */
  parentLabel?: string;
  parentTo?: { to: string; params?: Record<string, string> };
  /** Subtitle line rendered below the title. */
  subtitle?: React.ReactNode;
  /** Artwork thumbnail shown before the title. */
  artwork?: React.ReactNode;
  /** Right-aligned actions (buttons, etc.). */
  actions?: React.ReactNode;
  /** Extra className on the container (e.g. for blurred background). */
  className?: string;
  children?: React.ReactNode;
}

export default function PageHeader({
  title,
  parentLabel,
  parentTo,
  subtitle,
  artwork,
  actions,
  className = "",
  children,
}: PageHeaderProps) {
  return (
    <div className={`px-6 py-2 border-b border-border flex items-center gap-4 h-14 ${className}`}>
      {children}
      {parentLabel && parentTo && (
        <BackNav parentLabel={parentLabel} parentTo={parentTo} />
      )}
      {artwork}
      <div className="flex-1 min-w-0">
        <h2 className={`font-semibold truncate leading-snug ${subtitle ? "text-base" : "text-lg"}`}>{title}</h2>
        {subtitle && (
          <div className="flex items-center gap-3">
            {subtitle}
          </div>
        )}
      </div>
      {actions}
    </div>
  );
}
