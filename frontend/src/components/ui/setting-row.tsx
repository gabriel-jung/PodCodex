import { cn } from "@/lib/utils";

interface SettingRowProps {
  label: string;
  help?: string;
  children: React.ReactNode;
  /** Full-width content below the label/control row. */
  below?: React.ReactNode;
  className?: string;
}

export function SettingRow({ label, help, children, below, className }: SettingRowProps) {
  return (
    <div className={cn("py-3", className)}>
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <span className="text-sm font-medium">{label}</span>
          {help && <p className="text-xs text-muted-foreground mt-0.5">{help}</p>}
        </div>
        <div className="shrink-0">{children}</div>
      </div>
      {below && <div className="mt-3">{below}</div>}
    </div>
  );
}

interface SettingSectionProps {
  title: string;
  description?: string;
  children: React.ReactNode;
  className?: string;
}

export function SettingSection({ title, description, children, className }: SettingSectionProps) {
  return (
    <div className={cn("space-y-1", className)}>
      <h4 className="text-sm font-medium">{title}</h4>
      {description && <p className="text-xs text-muted-foreground">{description}</p>}
      <div className="divide-y divide-border/60">{children}</div>
    </div>
  );
}
