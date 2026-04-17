import { Fragment } from "react";
import { ChevronRight } from "lucide-react";

export interface BreadcrumbItem {
  label: string;
  onClick?: () => void;
}

interface PageHeaderProps {
  title: string;
  /** Subtitle line rendered below the title. */
  subtitle?: React.ReactNode;
  /** Breadcrumb trail rendered above the title (Home / Show / …). */
  breadcrumbs?: BreadcrumbItem[];
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
  subtitle,
  breadcrumbs,
  artwork,
  actions,
  className = "",
  children,
}: PageHeaderProps) {
  const hasCrumbs = breadcrumbs && breadcrumbs.length > 0;
  return (
    <div className={`px-6 py-2 border-b border-border flex items-center gap-4 ${hasCrumbs || subtitle ? "h-16" : "h-14"} ${className}`}>
      {children}
      {artwork}
      <div className="flex-1 min-w-0">
        {hasCrumbs && (
          <nav className="flex items-center gap-1 text-2xs text-muted-foreground mb-0.5 truncate">
            {breadcrumbs.map((b, i) => (
              <Fragment key={b.label}>
                {i > 0 && <ChevronRight className="w-3 h-3 shrink-0 opacity-60" />}
                {b.onClick ? (
                  <button
                    onClick={b.onClick}
                    className="hover:text-foreground transition truncate max-w-[14rem]"
                  >
                    {b.label}
                  </button>
                ) : (
                  <span className="truncate max-w-[14rem]">{b.label}</span>
                )}
              </Fragment>
            ))}
          </nav>
        )}
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
