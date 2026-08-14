import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Props {
  refreshing: boolean;
  onClick: () => void;
  title?: string;
  className?: string;
}

/** Icon-only ghost refresh button: spins and disables while refreshing.
 *  Shared by the settings/system panels (GPU, ffmpeg, Ollama). */
export function RefreshIconButton({ refreshing, onClick, title, className = "h-7" }: Props) {
  return (
    <Button variant="ghost" size="sm" onClick={onClick} disabled={refreshing} title={title} className={className}>
      <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? "animate-spin" : ""}`} />
    </Button>
  );
}
