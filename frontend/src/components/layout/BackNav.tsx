import { useNavigate } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ChevronRight } from "lucide-react";

/** Back arrow + parent breadcrumb, used in page headers.
 *  - Arrow: history.back() (or parent if no history)
 *  - Label: always navigates to parent page
 */
export default function BackNav({ parentLabel, parentTo }: {
  parentLabel: string;
  parentTo: { to: string; params?: Record<string, string> };
}) {
  const navigate = useNavigate();

  const goBack = () => {
    if (window.history.length > 1) {
      window.history.back();
    } else {
      navigate(parentTo as Parameters<typeof navigate>[0]);
    }
  };

  return (
    <div className="flex items-center border-r border-border pr-3 mr-1">
      <Button onClick={goBack} variant="ghost" size="icon" className="h-7 w-7 shrink-0" title="Back">
        <ArrowLeft className="w-4 h-4" />
      </Button>
      <ChevronRight className="w-3.5 h-3.5 text-muted-foreground/50 shrink-0" />
      <Button
        onClick={() => navigate(parentTo as Parameters<typeof navigate>[0])}
        variant="ghost"
        size="sm"
        className="text-muted-foreground px-1.5 h-7"
      >
        {parentLabel}
      </Button>
    </div>
  );
}
