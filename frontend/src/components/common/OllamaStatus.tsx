import { CheckCircle2, AlertCircle } from "lucide-react";
import { RefreshIconButton } from "@/components/common/RefreshIconButton";
import type { OllamaCheckResponse } from "@/api/types";

interface OllamaStatusProps {
  data: OllamaCheckResponse | undefined;
  isFetching: boolean;
  onRefresh: () => void;
}

export default function OllamaStatus({ data, isFetching, onRefresh }: OllamaStatusProps) {
  const reachable = data?.reachable ?? false;
  const modelCount = data?.models.length ?? 0;

  return (
    <div
      className={`flex items-start gap-2 rounded-md border px-3 py-2 text-xs ${
        reachable
          ? "border-success/30 bg-success/5"
          : "border-warning/30 bg-warning/5"
      }`}
    >
      {reachable ? (
        <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-success" />
      ) : (
        <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
      )}
      <div className="min-w-0 flex-1">
        {reachable ? (
          <p className="font-medium">
            {modelCount > 0
              ? `Connected, ${modelCount} model${modelCount === 1 ? "" : "s"}`
              : "Connected, no models pulled yet"}
          </p>
        ) : (
          <p>
            <span className="font-medium">Ollama not running.</span>{" "}
            <a
              href="https://ollama.com"
              target="_blank"
              rel="noreferrer"
              className="text-muted-foreground underline"
            >
              Install and open it
            </a>
            <span className="text-muted-foreground">, then refresh.</span>
          </p>
        )}
      </div>
      <RefreshIconButton refreshing={isFetching} onClick={onRefresh} className="h-7 px-2" />
    </div>
  );
}
