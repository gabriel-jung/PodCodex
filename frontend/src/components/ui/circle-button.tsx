import * as React from "react";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface CircleButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  icon: LucideIcon;
  variant?: "default" | "destructive" | "muted";
}

const variantClasses = {
  default: "text-muted-foreground hover:text-foreground hover:bg-accent",
  destructive: "text-muted-foreground hover:text-destructive hover:bg-destructive/10",
  muted: "text-muted-foreground/60 hover:text-muted-foreground hover:bg-muted",
};

const CircleButton = React.forwardRef<HTMLButtonElement, CircleButtonProps>(
  ({ icon: Icon, variant = "default", className, ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          "h-7 w-7 inline-flex items-center justify-center rounded-full transition-colors",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
          "disabled:pointer-events-none disabled:opacity-50",
          variantClasses[variant],
          className,
        )}
        {...props}
      >
        <Icon className="h-3.5 w-3.5" />
      </button>
    );
  },
);
CircleButton.displayName = "CircleButton";

export { CircleButton };
