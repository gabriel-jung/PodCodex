import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Download, Subtitles, ChevronDown } from "lucide-react";
import { languageToISO, SUB_LANGUAGES } from "@/lib/utils";

interface DownloadDropdownProps {
  isYouTube: boolean;
  showLanguage: string;
  onDownload: () => void;
  onImportSubs: (lang: string) => void;
  disabled: boolean;
  subsLabel: string;
  subsEnabled: boolean;
  audioLabel: string;
  showAudio: boolean;
  audioEnabled: boolean;
  variant?: "default" | "outline";
  size?: "sm" | "default";
  align?: "left" | "right";
  className?: string;
}

/** Sort subtitle languages, prioritizing the show's language. */
export function sortLanguagesByShow(showLanguage: string) {
  const showLangCode = languageToISO(showLanguage) || showLanguage.toLowerCase().slice(0, 2);
  return [...SUB_LANGUAGES].sort((a, b) => {
    if (a.code === showLangCode) return -1;
    if (b.code === showLangCode) return 1;
    return 0;
  });
}

export default function DownloadDropdown({
  isYouTube,
  showLanguage,
  onDownload,
  onImportSubs,
  disabled,
  subsLabel,
  subsEnabled,
  audioLabel,
  showAudio,
  audioEnabled,
  variant = "outline",
  size = "sm",
  align = "left",
  className = "",
}: DownloadDropdownProps) {
  const [open, setOpen] = useState(false);

  // Non-YouTube: simple download button
  if (!isYouTube) {
    if (!showAudio) return null;
    return (
      <Button
        onClick={onDownload}
        disabled={!audioEnabled || disabled}
        variant={variant}
        size={size}
        className={className}
      >
        <Download className="w-3 h-3 mr-1" /> Download
      </Button>
    );
  }

  const sortedLangs = sortLanguagesByShow(showLanguage);
  const showLangCode = languageToISO(showLanguage) || showLanguage.toLowerCase().slice(0, 2);
  const primaryLang = sortedLangs.find((l) => l.code === showLangCode) || sortedLangs[0];
  const close = () => setOpen(false);

  return (
    <div className="relative">
      <Button
        onClick={() => { if (open) close(); else setOpen(true); }}
        disabled={disabled}
        variant={variant}
        size={size}
        className={className}
      >
        <Download className="w-3 h-3" /> Download <ChevronDown className="w-3 h-3 ml-0.5" />
      </Button>
      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={close} />
          <div className={`absolute ${align === "right" ? "right-0" : "left-0"} top-full mt-1 z-50 bg-popover border border-border rounded-md shadow-lg py-1 min-w-[160px]`}>
            <button
              onClick={() => { close(); onImportSubs(primaryLang.code); }}
              disabled={!subsEnabled || disabled}
              className="w-full flex items-center gap-2 px-3 py-1.5 text-xs hover:bg-accent transition disabled:opacity-40"
            >
              <Subtitles className="w-3 h-3" /> {subsLabel}
            </button>
            {showAudio && (
              <>
                <div className="border-t border-border/50 my-0.5" />
                <button
                  onClick={() => { close(); onDownload(); }}
                  disabled={!audioEnabled || disabled}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-xs hover:bg-accent transition disabled:opacity-40"
                >
                  <Download className="w-3 h-3" /> {audioLabel}
                </button>
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}
