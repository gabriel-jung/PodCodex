import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
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

  const showLangCode = languageToISO(showLanguage) || showLanguage.toLowerCase().slice(0, 2);
  const primaryLang = SUB_LANGUAGES.find((l) => l.code === showLangCode) ?? SUB_LANGUAGES[0];

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild disabled={disabled}>
        <Button disabled={disabled} variant={variant} size={size} className={className}>
          <Download className="w-3 h-3" /> Download <ChevronDown className="w-3 h-3 ml-0.5" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align={align === "right" ? "end" : "start"} className="min-w-[160px]">
        <DropdownMenuItem
          onSelect={() => onImportSubs(primaryLang.code)}
          disabled={!subsEnabled || disabled}
          className="text-xs"
        >
          <Subtitles className="w-3 h-3" /> {subsLabel}
        </DropdownMenuItem>
        {showAudio && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onSelect={onDownload}
              disabled={!audioEnabled || disabled}
              className="text-xs"
            >
              <Download className="w-3 h-3" /> {audioLabel}
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
