# PodCodex Design System

> Warm editorial aesthetic for a podcast pipeline desktop app. Cream paper canvas with sepia ink and a gold-leaf accent. Reads like an oil-printed catalogue raisonné under tungsten light. Built on shadcn/ui + Tailwind 4 + React 19, packaged in a Tauri shell.

**Canonical references** (when in doubt, model new components on these):
- `frontend/src/components/index/IndexInspectorModal.tsx`: dense list dialog with mono timestamps, IDs, opacity-de-emphasized values
- `frontend/src/components/search/SegmentContextDialog.tsx`: same idiom, search-context variant
- `frontend/src/components/show/ShowProgressStrip.tsx`: text-only stage-hued ledger; reuse pattern for any per-stage breakdown
- `frontend/src/components/index/IndexRow.tsx`: shared collection row (whole-row click + optional `deletion` bundle)

**Single facilities** (never reroll inline; extend if missing):
- `frontend/src/lib/stageClasses.ts`: stage palette (`STAGE_CLASSES[stage]`: `bg`, `text`, `border`, `borderL`, `dot`)
- `frontend/src/lib/stepStatus.ts`: `PanelStatus` (`"ready"|"review"|"none"`) and its `PANEL_STATUS_META` labels, `reviewStatus` / `plainStatus` / `translationsStatus` / `isEdited`, and `episodeCanRunStep` (the one rule for which episodes can run a step)
- `frontend/src/lib/showCounts.ts`: episode count label formatting
- `frontend/src/lib/runRequests.ts`: `llmRunFields`, `transcribeRunFields` for every run request, episode panel and batch alike
- `frontend/src/lib/episodeRef.ts`: `getEpisodeSourceRef` for every per-episode query key, `getEpisodeBatchPath` for batch keys
- `frontend/src/lib/speakerColor.ts`: speaker hues; `frontend/src/lib/speakers.ts`: unattributed-speaker constants

## 1. Visual Theme & Atmosphere

PodCodex rejects the cold dark-mode-default of dev tooling. The canvas is **warm cream paper** in light mode (`oklch(0.962 0.012 78)`, a soft off-white with imperceptible orange undertone) and **warm near-black** in dark mode (`oklch(0.158 0.008 55)`, never pure black, always sepia-tinted). A subtle paper-grain SVG noise overlay sits above all content via `multiply` blend in light, `screen` in dark, at 5–7% opacity. The texture is felt, not seen; UIs feel printed, not rendered.

Typography pairs **Inter Variable** for UI/body with **Fraunces Variable** (variable serif) for headings (`h1`, `h2`). Inter carries OpenType features `cv11, ss01, ss03` globally for the cleaner geometric alternates; Fraunces uses `ss01, onum` (old-style numerals) at headings with `-0.015em` letter-spacing. **JetBrains Mono Variable** handles code, timestamps, and technical labels.

The accent is **gold-leaf** (`oklch(0.595 0.145 68)`, warm amber, not yellow), reserved for primary actions and active states. Secondary surfaces step up in warm-gray luminance rather than introducing color. Status colors (success green, warning amber, destructive red) are warm-shifted to coexist with the cream canvas.

**Key Characteristics:**
- Warm editorial canvas: cream `oklch(0.962 0.012 78)` light / sepia-black `oklch(0.158 0.008 55)` dark
- Paper-grain SVG noise overlay at 5–7% opacity (multiply light, screen dark)
- Inter Variable (`cv11, ss01, ss03`) for UI; Fraunces Variable (`ss01, onum`) for h1/h2
- Gold-leaf primary: `oklch(0.595 0.145 68)` light / `oklch(0.745 0.145 72)` dark
- All colors expressed in `oklch()` for perceptual uniformity
- shadcn/ui token model (`--background`, `--foreground`, `--card`, etc.); never raw hex in components
- Single base radius `--radius: 0.5rem` (8px); `sm/md/lg/xl` derive from it
- Two-column settings panels (label left, control right)
- Dense lists: zebra-free, hover-tinted only
- WCAG AA verified at the 5% grain opacity

## 2. Color Palette & Roles

All colors are CSS custom properties resolved via `@theme inline` to Tailwind utilities (`bg-background`, `text-muted-foreground`, etc.). **Never hardcode hex or oklch in components**; read tokens.

### Light mode (default)
| Token | Value | Role |
|-------|-------|------|
| `--background` | `oklch(0.962 0.012 78)` | Page canvas (cream paper) |
| `--foreground` | `oklch(0.205 0.018 50)` | Primary ink |
| `--card` | `oklch(0.970 0.011 78)` | Card surface (one shade brighter than bg) |
| `--popover` | `oklch(0.985 0.008 78)` | Popover/dropdown surface |
| `--primary` | `oklch(0.595 0.145 68)` | Gold-leaf accent: CTAs, focus ring |
| `--secondary` | `oklch(0.918 0.014 76)` | Muted surface: chips, inputs |
| `--muted` | `oklch(0.918 0.014 76)` | Same as secondary; disabled/recessed |
| `--muted-foreground` | `oklch(0.470 0.025 58)` | Metadata, captions, timestamps |
| `--accent` | `oklch(0.895 0.022 72)` | Warm-tinted hover surface |
| `--border` | `oklch(0.855 0.020 75)` | All separators and input outlines |
| `--destructive` | `oklch(0.530 0.185 30)` | Errors, delete actions |
| `--success` | `oklch(0.595 0.130 132)` | Completion, indexed status |
| `--warning` | `oklch(0.715 0.165 82)` | Pending, attention needed |
| `--info` | `oklch(0.595 0.130 240)` | In-progress, partial, changed (warm-shifted blue) |

### Dark mode (`.dark` class on `<html>`)
| Token | Value | Notes |
|-------|-------|-------|
| `--background` | `oklch(0.158 0.008 55)` | Sepia-black, never pure |
| `--foreground` | `oklch(0.935 0.013 80)` | Warm off-white, never `#fff` |
| `--card` | `oklch(0.190 0.009 55)` | One luminance step above bg |
| `--popover` | `oklch(0.220 0.010 55)` | Two steps above bg |
| `--primary` | `oklch(0.745 0.145 72)` | Brighter gold for dark contrast |
| `--secondary` | `oklch(0.265 0.011 55)` | Recessed surface |
| `--muted-foreground` | `oklch(0.625 0.020 68)` | Tertiary text |
| `--accent` | `oklch(0.285 0.018 62)` | Hover tint |
| `--border` | `oklch(0.275 0.012 55)` | Hairline separator |
| `--info` | `oklch(0.745 0.140 240)` | Brighter info-blue for dark contrast |

### Surface elevation model
PodCodex elevates via **luminance step**, not shadow. Each level lifts the background by ~0.02 in oklch L:

```
bg (canvas)  →  card  →  popover        (dialogs sit on bg-background, lifted by shadow-lg and the backdrop)
0.962        0.970    0.985        light
0.158        0.190    0.220        dark
```

## 3. Typography Rules

### Font Family
- **Sans (UI/body):** `Inter Variable, ui-sans-serif, system-ui, -apple-system, sans-serif`
- **Display (h1, h2):** `Fraunces Variable, ui-serif, Georgia, serif`
- **Mono (code, timestamps, technical labels):** `JetBrains Mono Variable, ui-monospace, SFMono-Regular, Menlo, monospace`

### OpenType features
- **Inter (global on `body`):** `"cv11", "ss01", "ss03"` (geometric alternates, single-story `a`, cleaner letterforms)
- **Fraunces (h1, h2):** `"ss01", "onum"` (old-style figures for editorial feel)

### Hierarchy

| Role | Font | Tailwind | Weight | Letter-spacing | Use |
|------|------|----------|--------|----------------|-----|
| Display | Fraunces | `text-3xl`–`text-5xl` | 500–600 | `-0.015em` | Hero, page titles |
| H1 | Fraunces | `text-3xl` | 600 | `-0.015em` | Page title (one per page, via `EditorialHeader`) |
| H2 (prose) | Fraunces | `text-2xl` | 600 | `-0.015em` | Section landmarks on prose pages |
| H2 (dense) | Inter | `text-base` | 600 | normal | Subsection in settings/forms (Fraunces stacked 8+ times reads as title duplication, so demote: drop `font-display`, shrink icons to `w-4 h-4`) |
| H3 | Inter | `text-xl` | 600 | normal | Card titles |
| H3 (dense) | Inter | `text-sm` | 600 | normal | Sub-subsection inside dense panels |
| H4 | Inter | `text-lg` | 600 | normal | Group headers |
| Body | Inter | `text-base` | 400 | normal | Paragraph text |
| Body-emphasis | Inter | `text-base` | 500 | normal | Inline strong |
| Small | Inter | `text-sm` | 400 | normal | Status and empty-state messages, form labels |
| Small-medium | Inter | `text-sm` | 500 | normal | Active nav, buttons |
| Caption | Inter | `text-xs` | 400 | normal | Metadata, episode date/duration, status |
| Tiny | Inter | `text-2xs` (`0.625rem`) | 500 | normal | Inline pills, version tags |
| Mono | JetBrains Mono | `font-mono text-xs`–`text-sm` | 400 | normal | Code, hashes, IDs, timestamps |

### Density tiers (canonical, derived from inspect dialogs)

| Tier | Size | Use |
|------|------|-----|
| Dense list row | `text-xs` | DirRow, FileRow, episode row metadata, inspect dialog rows, breadcrumb, sidebar nav buttons, footnotes |
| Status / empty / loading message | `text-sm` | Centered "Loading…", "No results", error blurbs, empty-state copy |
| Modal/panel header | `DialogTitle` default (`text-lg`), or `text-sm`/`text-base font-semibold` in dense dialogs | Header in any modal |
| Reading prose | `text-base` | Transcript view, long-form modal copy |
| Inline status pill | `text-2xs` (`0.625rem`) weight 500 | "show", "audio", "indexed" badges next to row labels |
| Log / debug block | `text-3xs` (`0.55rem`) `font-mono` | `<pre>` log viewers, debug detail collapsibles |

### Principles
- **Default to `text-xs` for dense panels.** Lists, settings rows, breadcrumbs, sidebars. `text-sm` is for centered status/empty messages, not for list items.
- **Serif for headings only.** Don't use Fraunces in body, captions, or buttons. Fraunces is for landmarks.
- **No all-caps. No `tracking-wider`/`tracking-widest`.** Both feel mid-2010s SaaS. Use weight 500–600 for emphasis instead.
- **Max font weight is 600 (`font-semibold`).** Never `font-bold` (700), `font-extrabold` (800), or `font-black` (900). Inter at warm-editorial cream looks aggressive past 600.
- **Mono for content with structure.** Timestamps, hashes, model IDs, file paths, code. Always pair with `tabular-nums` when numerals matter (timestamps, durations, byte sizes).
- **De-emphasize via opacity, not color shift.** Inspect dialogs use `text-muted-foreground/60` (60% opacity on already-muted) for tertiary values. Avoid inventing a fourth gray.
- **Numerals in Fraunces use `onum`** (old-style); Inter uses lining figures by default; let it.

## 4. Component Stylings

### Buttons (shadcn `<Button />` variants)
| Variant | Bg | Text | Border | Use |
|---------|-----|------|--------|-----|
| `default` (primary) | `bg-primary` | `text-primary-foreground` | none | Primary CTA per view (one max) |
| `secondary` | `bg-secondary` | `text-secondary-foreground` | none | Standard action |
| `outline` | transparent | `text-foreground` | `border-border` | Tertiary action, modal cancels |
| `ghost` | transparent | `text-foreground` | none | Toolbar, icon buttons, dense rows |
| `destructive` | `bg-destructive` | `text-destructive-foreground` | none | Delete, irreversible |
| `link` | none | `text-primary` | none | Inline navigation |

- Radius: `rounded-md` (6px) default, `rounded-lg` (8px) for prominent CTAs, `rounded-full` for icon-only circular toggles
- Padding: `px-3 py-1.5` small, `px-4 py-2` standard, `px-6 py-3` large
- Focus: `ring-2 ring-ring ring-offset-2 ring-offset-background` (gold-leaf ring)

### Cards
- Background: `bg-card`
- Border: `border border-border`
- Radius: `rounded-lg` (8px)
- Padding: `p-4` (compact), `p-6` (standard)
- Shadow: none by default; `shadow-sm` only on floating popovers
- Hover (when interactive): `hover:border-muted-foreground/30` (border tint); never elevate via shadow, translate or scale. Rows use `hover:bg-accent`.

### Inputs
Use the `.input` utility from `index.css`:
```
@apply w-full bg-secondary text-sm text-foreground rounded-md px-2 py-1
       border border-border focus:border-primary focus:outline-none
       placeholder:text-muted-foreground;
```
- Background is `bg-secondary` (not `bg-background`); inputs sit recessed against canvas
- Focus replaces border color rather than adding ring (subtler)

### Lists & rows (episode rows, inspect rows, settings rows)
- Body: `text-xs` (inspect-dialog standard); metadata aligned right
- Inline row container: `rounded` (4px), not `rounded-md`. Buttons stay `rounded-md`; rows stay `rounded`.
- Hover: `hover:bg-accent` (warm tint)
- No alternating zebra stripes; borders or no separator at all
- Active/selected: `bg-accent` persistent + `text-foreground`
- Mono columns (timestamps, IDs, hashes): `font-mono tabular-nums`; de-emphasize with `text-muted-foreground/60` rather than introducing a new color
- Status pills inside rows: `text-2xs` weight 500, `bg-{role}/15 text-{role}` (e.g. `bg-primary/15 text-primary` for "show", `bg-warning/15 text-warning` for "audio"). Never `bg-success text-white`.

### Badges / pills
- Status (indexed, pending, error): inline dot or filled pill
- Use `bg-success/15 text-success` style (color at 15% alpha bg, full color text); don't fill
- Radius: `rounded-full` for pills, `rounded` (4px) for inline tags
- Font: `text-2xs` (`0.625rem`) weight 500; never `tracking-wider`

### Icon ↔ text size pairing (warnings, status, badges)
Match icon dimension to its accompanying text size; icon bigger than text reads as top-heavy.

| Tier | Use | Text | Icon |
|------|-----|------|------|
| Inline badge | "1 unnamed", status pill in row | `text-2xs` | `w-2.5 h-2.5` |
| Callout / toolbar button | banner inside panel, flagged-toggle button | `text-xs` | `w-3 h-3` |
| Dialog / modal heading | error modal, warning dialog | `text-sm` | `w-4 h-4` |

Don't pair `text-2xs` with `w-3.5 h-3.5` icons or `text-xs` with `w-4 h-4` icons (visual mismatch).

### Modals / dialogs
- **Prefer shadcn `<Dialog>` primitive.** `IndexInspectorModal` and `SegmentContextDialog` are the canonical examples. Custom modal divs are tolerated but should migrate when touched.
- Title: `DialogTitle` renders an `h2`, so it is Fraunces through the global heading rule; its default is `text-lg font-semibold`. Size it down with a class (`text-sm` / `text-base font-semibold`) in dense dialogs
- Description: `text-sm text-muted-foreground` (shadcn `DialogDescription` default)
- Body density: `text-xs` for dense rows; `text-sm` for prose, status, and empty states
- Surface: `bg-background` (the `ui/dialog` default; the ring of `shadow-lg` and the backdrop separate it)
- Border: `border border-border`
- Radius: `rounded-lg` (never `rounded-xl`)
- Shadow: `shadow-lg` (never `shadow-2xl`)
- Backdrop: `bg-black/50` (shadcn `DialogOverlay` default; match across custom modals too)
- Two-column layout for settings: `FormGrid` (`sm:grid-cols-[auto_1fr]`), label column sized to content, control column takes the rest

### Audio player (`AudioBar`)
- **No waveform.** Seek bar is a simple horizontal track + thumb. Waveform visualizations are decorative noise here.
- Track: `bg-muted` 4px tall, gold-leaf fill `bg-primary` for played portion
- Time displays in `font-mono text-xs`

## 5. Layout Principles

### Spacing scale (Tailwind defaults; document the rhythm we actually hit)
- Tight rhythm: `gap-1` (4px), `gap-1.5` (6px), `gap-2` (8px); within rows, between icon+label
- Standard rhythm: `gap-3` (12px), `gap-4` (16px); within cards
- Section rhythm: `gap-6` (24px), `gap-8` (32px); between cards, panels
- Page rhythm: `py-8`, `py-12`; page padding

### Grid
- Sidebar (left nav): `w-14` collapsed, `w-48` expanded; fixed, dense. The width comes from `lib/sidebar.ts` (shared with the content padding in `RootLayout`).
- Main column: flexible; full width for lists and the transcript editor; `max-w-3xl` for the settings page
- Settings: `FormGrid`, two columns from `sm:`, single column below
- Episode lists: full-width single column with right-aligned metadata
- Card grids (library, episode tiles): `repeat(auto-fill, minmax(<min>, 1fr))`, NOT `repeat(N, 1fr)`. Wider viewports add columns once a full new column fits; leftover width between thresholds is absorbed by mild stretch so the grid never leaves an empty right-edge gap. Plain `repeat(N, 1fr)` (no min) lets cards grow into billboards on wide screens, which the user explicitly rejected.

### Border radius scale
- `rounded` (4px): inline list rows, tags, badges, micro-elements
- `rounded-md` (6px): buttons, inputs, small cards
- `rounded-lg` (8px): cards, modals, popovers (default for all dialogs)
- `rounded-xl` (12px): *avoid by default*; reserved for featured marketing modules, not standard cards or modals
- `rounded-full`: circular icon buttons, status dots, pill chips, badges

## 6. Depth & Elevation

PodCodex uses **luminance step**, not box-shadow, for elevation. On the cream canvas, traditional drop shadows look smoke-grey and break the printed-paper feel. On the sepia-black dark canvas, dark-on-dark shadows are invisible anyway.

| Level | Treatment | Use |
|-------|-----------|-----|
| Canvas (0) | `bg-background` | Page background |
| Surface (1) | `bg-card` + `border-border` | Cards, panels |
| Floating (2) | `bg-popover` + `border-border` + `shadow-sm` | Dropdowns, popovers, tooltips (`ui/dropdown-menu`, `ui/popover`) |
| Dialog (3) | `bg-background` + `border-border` + `shadow-lg` over a `bg-black/50` backdrop | Modals only, always through `ui/dialog` |

Shadows only on floating layers (`shadow-sm`) and dialogs (`shadow-lg`, the cap). Never `shadow-xl` or `shadow-2xl`. Decorative `shadow-lg` on overlay buttons (e.g. play/download hover overlays on cards) is forbidden; use border or background tint instead.

## 7. Rules of thumb

- Omit defaults and absent values in labels (don't render "Speaker: -" when there's no speaker)
- No pure white (`#ffffff`) or pure black (`#000000`) on theme surfaces; both fight the warm canvas (media scrims below are the exception)
- Gold-leaf is the only accent. Stage hues (`STAGE_CLASSES`) and speaker hues (`lib/speakerColor.ts`) are identity colors, not accents, and stay inside their facilities.
- No em dashes in UI copy; use commas, colons, periods, or parentheses
- Don't disable the paper-grain overlay; it's load-bearing for the editorial feel

### Pipeline stage palette
Pipeline stages get distinct hues for quick recognition. Tokens defined in `index.css`, warm-shifted to coexist with the cream + gold-leaf canvas:

| Stage | Token | Utility | Hue |
|-------|-------|---------|-----|
| Transcribed | `--stage-transcribe` | `bg-stage-transcribe`, `text-stage-transcribe` | 240° (blue) |
| Corrected | `--stage-correct` | `bg-stage-correct`, `text-stage-correct` | 305° (purple) |
| Translated | `--stage-translate` | `bg-stage-translate`, `text-stage-translate` | 195° (teal) |
| Synthesized | `--stage-synth` | `bg-stage-synth`, `text-stage-synth` | 50° (orange) |
| Indexed | `--warning` | `bg-warning/15 text-warning` | 82° (amber, reuses semantic token) |

Each stage chip uses the `bg-{stage}/15 text-{stage}` pattern. **Always read tokens through `STAGE_CLASSES` from `lib/stageClasses.ts`**; never hand-roll `text-stage-*` strings in components. If new stages are added, extend `index.css`, `STAGE_CLASSES`, and this table; don't reach for raw Tailwind colors.

Speaker colors come from `lib/speakerColor.ts`: eleven fixed hues at `oklch(0.62 0.17 h)`, picked by a stable hash of the name. Never pick a speaker color anywhere else.

### Pipeline status vocabulary
Three review states are surfaced consistently across StageCard (Overview hub), `PipelinePanel` headers, and `StatusChips` suffix copy. Derive via `reviewStatus` / `plainStatus` / `translationsStatus` from `lib/stepStatus.ts`; never hand-roll the present+edited check.

| State | Label | Color | Meaning |
|-------|-------|-------|---------|
| `ready` | "ready" | `text-success` | Output present and edited (`isEdited` true) |
| `review` | "needs review" | `text-info` | Output present but raw (awaiting human review) |
| `none` | "not started" | `text-muted-foreground/60` | No output yet |

Synth + index have no review concept; `plainStatus` collapses them to `ready` / `none`.

### Exception: media scrims
Elements rendered **on top of artwork or video** (episode card overlays, play buttons, image badges) need contrast against arbitrary image content, not against the theme. The token system would flip in dark mode and break legibility on bright images. Allowed only on top of media:
- `bg-black/{60,65}` and `bg-white/{90,95}` for scrims and floating buttons
- `text-white`, `text-black` paired with the scrims
- `bg-gradient-to-t from-black/60 to-transparent` for bottom-fade overlays
- `shadow-lg` on floating media buttons (play/download): legibility, not decoration

These never appear on theme surfaces. If you find one on a card or panel without an image behind it, replace with tokens.

## 8. Window sizes and targets

A desktop window, not a phone. Tailwind's default breakpoints apply; the one that matters is `sm` (≥640px), where `FormGrid` switches to two columns. The sidebar collapses to icons by user toggle (persisted in `layoutStore`), not by breakpoint; icon buttons carry the label as a title.

- Minimum 32×32 hit area for icon buttons (`p-2` on a `w-4 h-4` icon)
- Pill chips: `px-2.5 py-1` minimum; episode rows: `py-2` minimum

## 9. Adding UI

1. shadcn/ui primitives from `frontend/src/components/ui/` first; custom only when the primitive doesn't fit
2. Tokens only: `bg-card`, `text-muted-foreground`, `border-border`, `bg-primary`
3. `text-xs` for dense lists and settings, `text-sm` for status and empty states, `text-base` for prose. `rounded-md` for inputs/buttons, `rounded-lg` for cards and dialogs
4. One `bg-primary` CTA per view; everything else `secondary`, `outline`, or `ghost`
5. If it looks "AI-generic SaaS", check for uppercase labels, `tracking-wider`, gradient buttons, shadows on cards, a waveform in the audio player. If colors feel cold, the warm hue (`oklch H ≈ 55–80`) was lost

## 10. Episode hub conventions

Single-consumer patterns introduced by the Episode → Overview tab. Read the components in `frontend/src/pages/EpisodePage.tsx` for the actual classes; the rules below are the *why*, not the CSS.

- **Stage card** (`StageCard`): represents a pipeline stage as a clickable card. Stage hue used for the icon tile and a `border-l-2` accent when content exists; empty stages use a dashed border and no accent. The card itself is the hit target, no inline "Open" CTA. Click opens the read-only viewer when content exists, otherwise navigates to the stage editor.
- **Activity log row** (`ActivityLog`): two-line clickable row: stage chip + `versionLabel` on top, `versionDate · {n} seg [· edited]` underneath. Caller pre-sorts desc by timestamp; component slices to a small cap. Click opens the viewer.
- **Flat versions table** (`VersionsTable`): when a hub aggregates artifacts from multiple stages, prefer one flat table over per-stage accordions. Plain-language headers ("Made with", not "Model · chain"). Whole row opens the viewer; only a hover-revealed trash icon stays on the row, with `e.stopPropagation()` so the row click still fires.
- **Click semantics across the hub:** stage cards, version rows, and activity rows all open `SegmentContextDialog` (read-only viewer). The viewer exposes "Open editor" so the editor surface is one extra click away. Don't add inline editor links to the hub.
- **Don't double-encode review state:** when a row already shows an `edited` marker (`text-2xs text-success`), keep the adjacent stage dot in the stage hue. Two greens means two signals, not one.

Promote to shared utilities when a second hub adopts these.

## 11. Per-episode panel state

Pipeline panels (`TranscribePanel`, `SynthesizePanel`, etc.) carry per-episode local state that must not bleed across episodes.

- **Remount on episode switch.** `EpisodePage` mounts each step's component with `<Panel key={"${step}|${episode.id}"} />`. The composite key forces React to unmount and re-init when either the active step or the active episode changes. Don't try to sync this with effects; the key is the single source of truth.
- **Parent-owned refs survive subcomponent unmount.** If part of a panel hides while a background job runs (e.g. the source-segment picker while a synth task runs), state local to that part is lost on remount. Keep it in a `useRef` in the panel above it. `SynthesizePanel`'s `sourceSelectionStampRef` is the canonical example. Panels themselves take no props and remount on episode switch, so nothing survives that by design.
- **Source-aware queries.** Synth queries (`voice-samples`, `generated-segments`, `versions`, `status`) cache-key on the value returned by `getEpisodeSourceRef` (`frontend/src/lib/episodeRef.ts`): `audio_path` when present, else `output_dir`. YouTube subtitle-only episodes have no source audio, so keying on `audio_path` alone makes their cache invalidation silently miss. Always destructure through `getEpisodeSourceRef`; never reach for `episode.audio_path` directly in a query key.
