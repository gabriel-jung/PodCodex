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
  /** Label for the subtitle option (e.g. "Subtitles (3 new)" or "Import subtitles") */
  subsLabel: string;
  /** Whether the subtitle option is enabled */
  subsEnabled: boolean;
  /** Label for the audio download button (e.g. "Audio (5)" or "Download audio") */
  audioLabel: string;
  /** Whether the audio download option is shown */
  showAudio: boolean;
  /** Whether the audio download option is enabled (within the dropdown) */
  audioEnabled: boolean;
  /** Button variant when the dropdown is not expanded */
  variant?: "default" | "outline";
  /** Button size */
  size?: "sm" | "default";
  /** Alignment of the dropdown menu */
  align?: "left" | "right";
  /** Extra button class */
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
  const [subsExpanded, setSubsExpanded] = useState(false);

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
  const close = () => { setOpen(false); setSubsExpanded(false); };

  return (
    <div className="relative">
      <Button
        onClick={() => { if (open) close(); else setOpen(true); }}
        disabled={disabled}
        variant={variant}
        size={size}
        className={className}
      >
        <Download className="w-3 h-3 mr-1" /> Download <ChevronDown className="w-3 h-3 ml-1" />
      </Button>
      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={close} />
          <div className={`absolute ${align === "right" ? "right-0" : "left-0"} top-full mt-1 z-50 bg-popover border border-border rounded-md shadow-lg py-1 min-w-[180px]`}>
            <button
              onClick={() => setSubsExpanded(!subsExpanded)}
              disabled={!subsEnabled || disabled}
              className="w-full flex items-center gap-2 px-3 py-1.5 text-xs hover:bg-accent transition disabled:opacity-40"
            >
              <Subtitles className="w-3 h-3" />
              {subsLabel}
              <ChevronDown className={`w-3 h-3 ml-auto transition ${subsExpanded ? "rotate-180" : ""}`} />
            </button>
            {subsExpanded && (
              <div className="border-t border-border/50 py-0.5">
                {sortedLangs.map((l) => (
                  <button
                    key={l.code}
                    onClick={() => { close(); onImportSubs(l.code); }}
                    className="w-full flex items-center gap-2 px-5 py-1 text-xs hover:bg-accent transition"
                  >
                    <span className="text-muted-foreground w-5">{l.code}</span> {l.label}
                  </button>
                ))}
              </div>
            )}
            {showAudio && (
              <button
                onClick={() => { close(); onDownload(); }}
                disabled={!audioEnabled || disabled}
                className="w-full flex items-center gap-2 px-3 py-1.5 text-xs hover:bg-accent transition disabled:opacity-40"
              >
                <Download className="w-3 h-3" /> {audioLabel}
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
}
