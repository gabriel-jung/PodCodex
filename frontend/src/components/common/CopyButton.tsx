import { useEffect, useState } from "react";
import { Check, Copy } from "lucide-react";

interface CopyButtonProps {
  value: string;
  className?: string;
  title?: string;
}

/** Tiny clipboard-copy button. 1.5s checkmark feedback after success. */
export default function CopyButton({ value, className, title = "Copy" }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const t = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(t);
  }, [copied]);
  return (
    <button
      type="button"
      onClick={async (e) => {
        e.stopPropagation();
        if (!value) return;
        try {
          await navigator.clipboard.writeText(value);
          setCopied(true);
        } catch {
          /* clipboard denied; nothing actionable to surface */
        }
      }}
      disabled={!value}
      className={`inline-flex items-center justify-center text-muted-foreground hover:text-foreground transition disabled:opacity-50 ${className ?? ""}`}
      title={copied ? "Copied" : title}
      aria-label={title}
    >
      {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
    </button>
  );
}
