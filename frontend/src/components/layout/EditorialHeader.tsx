/** Editorial page hero — large artwork, display title, typographic stats,
 *  optional blurred backdrop. Shared by Home, Show, and Episode pages. */

import { Fragment } from "react";
import { ChevronRight } from "lucide-react";
import type { LucideIcon } from "lucide-react";

export interface Crumb {
  label: string;
  onClick?: () => void;
}

export interface Stat {
  value: string | number | React.ReactNode;
  label?: string;
}

interface Props {
  title: string;
  subtitle?: string;
  breadcrumbs?: Crumb[];
  /** URL for both the artwork card and the blurred backdrop. */
  artworkUrl?: string;
  /** Fallback icon when no artwork is provided. */
  fallbackIcon?: LucideIcon;
  /** Click handler on the artwork (e.g. play audio). */
  onArtworkClick?: () => void;
  /** Overlay rendered on the artwork on hover (e.g. play icon). */
  artworkOverlay?: React.ReactNode;
  stats?: Stat[];
  /** Extra node rendered next to the stats (e.g. pipeline status pills). */
  statusSlot?: React.ReactNode;
  actions?: React.ReactNode;
}

export default function EditorialHeader({
  title,
  subtitle,
  breadcrumbs,
  artworkUrl,
  fallbackIcon: FallbackIcon,
  onArtworkClick,
  artworkOverlay,
  stats,
  statusSlot,
  actions,
}: Props) {
  const hasCrumbs = breadcrumbs && breadcrumbs.length > 0;
  const Artwork = onArtworkClick ? "button" : "div";

  return (
    <div className="relative border-b border-border overflow-hidden shrink-0">
      {artworkUrl && (
        <>
          <div
            aria-hidden
            className="absolute inset-0 bg-cover bg-center opacity-30 blur-3xl scale-110"
            style={{ backgroundImage: `url(${artworkUrl})` }}
          />
          <div
            aria-hidden
            className="absolute inset-0 bg-gradient-to-b from-background/60 via-background/80 to-background"
          />
        </>
      )}
      <div className="relative px-6 py-3 flex gap-4 items-center">
        {(artworkUrl || FallbackIcon) && (
          <Artwork
            onClick={onArtworkClick}
            className={`relative w-16 h-16 rounded-lg shrink-0 overflow-hidden ${
              onArtworkClick ? "group cursor-pointer" : ""
            } ${artworkUrl ? "shadow-md shadow-black/20 ring-1 ring-border/50" : "bg-muted"}`}
          >
            {artworkUrl ? (
              <img src={artworkUrl} alt={title} className="w-full h-full object-cover" />
            ) : FallbackIcon ? (
              <span className="w-full h-full flex items-center justify-center">
                <FallbackIcon className="w-7 h-7 text-muted-foreground/50" />
              </span>
            ) : null}
            {artworkOverlay && (
              <span className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 group-hover:opacity-100 transition">
                {artworkOverlay}
              </span>
            )}
          </Artwork>
        )}
        <div className="flex-1 min-w-0 flex flex-col gap-1">
          <div className="text-2xs text-muted-foreground truncate min-h-4 flex items-center gap-1">
            {hasCrumbs
              ? breadcrumbs!.map((c, i) => (
                  <Fragment key={i}>
                    {i > 0 && <ChevronRight className="w-3 h-3 shrink-0 opacity-60" />}
                    {c.onClick ? (
                      <button
                        onClick={c.onClick}
                        className="hover:text-foreground transition truncate max-w-[14rem]"
                      >
                        {c.label}
                      </button>
                    ) : (
                      <span className="truncate max-w-[14rem] text-foreground/70">{c.label}</span>
                    )}
                  </Fragment>
                ))
              : subtitle && <span className="truncate">{subtitle}</span>}
          </div>
          <div className="flex items-center gap-4 min-w-0">
            <h1
              className="font-display text-2xl font-semibold leading-tight truncate"
              title={title}
            >
              {title}
            </h1>
            {stats && stats.length > 0 && (
              <dl className="flex gap-4 text-xs shrink-0">
                {stats.map((s, i) => (
                  <div key={i} className="flex items-baseline gap-1">
                    <dt className="font-mono font-medium tabular-nums">{s.value}</dt>
                    {s.label && <dd className="text-muted-foreground">{s.label}</dd>}
                  </div>
                ))}
              </dl>
            )}
            {statusSlot && <div className="shrink-0">{statusSlot}</div>}
          </div>
        </div>
        {actions && <div className="shrink-0">{actions}</div>}
      </div>
    </div>
  );
}
