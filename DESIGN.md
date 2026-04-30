# PodCodex Design System

> Warm editorial aesthetic for a podcast pipeline desktop app. Cream paper canvas with sepia ink and a gold-leaf accent. Reads like an oil-printed catalogue raisonné under tungsten light. Built on shadcn/ui + Tailwind 4 + React 19, packaged in a Tauri shell.

**Canonical references** — when in doubt, model new components on these:
- `frontend/src/components/index/IndexInspectorModal.tsx` — dense list dialog with mono timestamps, IDs, opacity-de-emphasized values
- `frontend/src/components/search/SegmentContextDialog.tsx` — same idiom, search-context variant

## 1. Visual Theme & Atmosphere

PodCodex rejects the cold dark-mode-default of dev tooling. The canvas is **warm cream paper** in light mode (`oklch(0.962 0.012 78)` — a soft off-white with imperceptible orange undertone) and **warm near-black** in dark mode (`oklch(0.158 0.008 55)` — never pure black, always sepia-tinted). A subtle paper-grain SVG noise overlay sits above all content via `multiply` blend in light, `screen` in dark, at 5–7% opacity. The texture is felt, not seen — UIs feel printed, not rendered.

Typography pairs **Inter Variable** for UI/body with **Fraunces Variable** (variable serif) for headings (`h1`, `h2`). Inter carries OpenType features `cv11, ss01, ss03` globally for the cleaner geometric alternates; Fraunces uses `ss01, onum` (old-style numerals) at headings with `-0.015em` letter-spacing. **JetBrains Mono Variable** handles code, timestamps, and technical labels.

The accent is **gold-leaf** (`oklch(0.595 0.145 68)` — warm amber, not yellow), reserved for primary actions and active states. Secondary surfaces step up in warm-gray luminance rather than introducing color. Status colors (success green, warning amber, destructive red) are warm-shifted to coexist with the cream canvas.

**Key Characteristics:**
- Warm editorial canvas: cream `oklch(0.962 0.012 78)` light / sepia-black `oklch(0.158 0.008 55)` dark
- Paper-grain SVG noise overlay at 5–7% opacity (multiply light, screen dark)
- Inter Variable (`cv11, ss01, ss03`) for UI; Fraunces Variable (`ss01, onum`) for h1/h2
- Gold-leaf primary: `oklch(0.595 0.145 68)` light / `oklch(0.745 0.145 72)` dark
- All colors expressed in `oklch()` for perceptual uniformity
- shadcn/ui token model (`--background`, `--foreground`, `--card`, etc.) — never raw hex in components
- Single base radius `--radius: 0.5rem` (8px); `sm/md/lg/xl` derive from it
- Two-column settings panels (label left, control right)
- Dense lists: zebra-free, hover-tinted only
- WCAG AA verified at the 5% grain opacity

## 2. Color Palette & Roles

All colors are CSS custom properties resolved via `@theme inline` to Tailwind utilities (`bg-background`, `text-muted-foreground`, etc.). **Never hardcode hex or oklch in components** — read tokens.

### Light mode (default)
| Token | Value | Role |
|-------|-------|------|
| `--background` | `oklch(0.962 0.012 78)` | Page canvas (cream paper) |
| `--foreground` | `oklch(0.205 0.018 50)` | Primary ink |
| `--card` | `oklch(0.970 0.011 78)` | Card surface (one shade brighter than bg) |
| `--popover` | `oklch(0.985 0.008 78)` | Popover/dropdown surface |
| `--primary` | `oklch(0.595 0.145 68)` | Gold-leaf accent — CTAs, focus ring |
| `--secondary` | `oklch(0.918 0.014 76)` | Muted surface — chips, inputs |
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
bg (canvas)  →  card  →  popover  →  (elevated dialog uses popover + ring shadow)
0.962        0.970    0.985        light
0.158        0.190    0.220        dark
```

## 3. Typography Rules

### Font Family
- **Sans (UI/body):** `Inter Variable, ui-sans-serif, system-ui, -apple-system, sans-serif`
- **Display (h1, h2):** `Fraunces Variable, ui-serif, Georgia, serif`
- **Mono (code, timestamps, technical labels):** `JetBrains Mono Variable, ui-monospace, SFMono-Regular, Menlo, monospace`

### OpenType features
- **Inter (global on `body`):** `"cv11", "ss01", "ss03"` — geometric alternates, single-story `a`, cleaner letterforms
- **Fraunces (h1, h2):** `"ss01", "onum"` — old-style figures for editorial feel

### Hierarchy

| Role | Font | Tailwind | Weight | Letter-spacing | Use |
|------|------|----------|--------|----------------|-----|
| Display | Fraunces | `text-3xl`–`text-5xl` | 500–600 | `-0.015em` | Hero, page titles |
| H1 | Fraunces | `text-3xl` | 600 | `-0.015em` | Section landmarks |
| H2 | Fraunces | `text-2xl` | 600 | `-0.015em` | Subsection |
| H3 | Inter | `text-xl` | 600 | normal | Card titles, modal headers |
| H4 | Inter | `text-lg` | 600 | normal | Group headers |
| Body | Inter | `text-base` | 400 | normal | Paragraph text |
| Body-emphasis | Inter | `text-base` | 500 | normal | Inline strong |
| Small | Inter | `text-sm` | 400 | normal | Default for dense panels (lists, settings) |
| Small-medium | Inter | `text-sm` | 500 | normal | Active nav, buttons |
| Caption | Inter | `text-xs` | 400 | normal | Metadata, episode date/duration, status |
| Tiny | Inter | `text-2xs` (`0.625rem`) | 500 | normal | Inline pills, version tags |
| Mono | JetBrains Mono | `font-mono text-xs`–`text-sm` | 400 | normal | Code, hashes, IDs, timestamps |

### Density tiers (canonical, derived from inspect dialogs)

| Tier | Size | Use |
|------|------|-----|
| Dense list row | `text-xs` | DirRow, FileRow, episode row metadata, inspect dialog rows, breadcrumb, sidebar nav buttons, footnotes |
| Status / empty / loading message | `text-sm` | Centered "Loading…", "No results", error blurbs, empty-state copy |
| Modal/panel header | `text-sm font-semibold` (custom) or shadcn `DialogTitle` default | Header in custom modal; never override `DialogTitle` in shadcn dialog |
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
- **Numerals in Fraunces use `onum`** (old-style); Inter uses lining figures by default — let it.

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
- Hover (when interactive): `hover:bg-accent` — never elevate via shadow

### Inputs
Use the `.input` utility from `index.css`:
```
@apply w-full bg-secondary text-sm text-foreground rounded-md px-2 py-1
       border border-border focus:border-primary focus:outline-none
       placeholder:text-muted-foreground;
```
- Background is `bg-secondary` (not `bg-background`) — inputs sit recessed against canvas
- Focus replaces border color rather than adding ring (subtler)

### Lists & rows (episode rows, inspect rows, settings rows)
- Body: `text-xs` (inspect-dialog standard); metadata aligned right
- Inline row container: `rounded` (4px) — not `rounded-md`. Buttons stay `rounded-md`; rows stay `rounded`.
- Hover: `hover:bg-accent` (warm tint)
- No alternating zebra stripes — borders or no separator at all
- Active/selected: `bg-accent` persistent + `text-foreground`
- Mono columns (timestamps, IDs, hashes): `font-mono tabular-nums`; de-emphasize with `text-muted-foreground/60` rather than introducing a new color
- Status pills inside rows: `text-2xs` weight 500, `bg-{role}/20 text-{role}` (e.g. `bg-primary/20 text-primary` for "show", `bg-warning/20 text-warning` for "audio"). Never `bg-success text-white`.

### Badges / pills
- Status (indexed, pending, error): inline dot or filled pill
- Use `bg-success/15 text-success` style (color at 15% alpha bg, full color text) — don't fill
- Radius: `rounded-full` for pills, `rounded` (4px) for inline tags
- Font: `text-2xs` (`0.625rem`) weight 500; never `tracking-wider`

### Icon ↔ text size pairing (warnings, status, badges)
Match icon dimension to its accompanying text size — icon bigger than text reads as top-heavy.

| Tier | Use | Text | Icon |
|------|-----|------|------|
| Inline badge | "1 unnamed", status pill in row | `text-2xs` | `w-2.5 h-2.5` |
| Callout / toolbar button | banner inside panel, flagged-toggle button | `text-xs` | `w-3 h-3` |
| Dialog / modal heading | error modal, warning dialog | `text-sm` | `w-4 h-4` |

Don't pair `text-2xs` with `w-3.5 h-3.5` icons or `text-xs` with `w-4 h-4` icons — visual mismatch.

### Modals / dialogs
- **Prefer shadcn `<Dialog>` primitive** — `IndexInspectorModal` and `SegmentContextDialog` are the canonical examples. Custom modal divs are tolerated but should migrate when touched.
- Title: `text-lg font-semibold` (shadcn `DialogTitle` default — don't override)
- Description: `text-sm text-muted-foreground` (shadcn `DialogDescription` default)
- Body density: `text-xs` for dense rows; `text-sm` for prose, status, and empty states
- Surface: `bg-popover` (one luminance step above card)
- Border: `border border-border`
- Radius: `rounded-lg` (never `rounded-xl`)
- Shadow: `shadow-lg` (never `shadow-2xl`)
- Backdrop: `bg-black/50` (shadcn `DialogOverlay` default — match across custom modals too)
- Two-column layout for settings: label column 1/3, control column 2/3

### Audio player (`AudioBar`)
- **No waveform.** Seek bar is a simple horizontal track + thumb. Waveform visualizations are decorative noise here.
- Track: `bg-muted` 4px tall, gold-leaf fill `bg-primary` for played portion
- Time displays in `font-mono text-xs`

## 5. Layout Principles

### Spacing scale (Tailwind defaults; document the rhythm we actually hit)
- Tight rhythm: `gap-1` (4px), `gap-1.5` (6px), `gap-2` (8px) — within rows, between icon+label
- Standard rhythm: `gap-3` (12px), `gap-4` (16px) — within cards
- Section rhythm: `gap-6` (24px), `gap-8` (32px) — between cards, panels
- Page rhythm: `py-8`, `py-12` — page padding

### Grid
- Sidebar (left nav): `w-56` to `w-64` — fixed, dense
- Main column: flexible, `max-w-5xl` for content-heavy pages (transcript reader); full-width for data tables
- Settings: two columns at `md:` breakpoint, single column on mobile
- Episode lists: full-width single column with right-aligned metadata

### Border radius scale
- `rounded` (4px) — inline list rows, tags, badges, micro-elements
- `rounded-md` (6px) — buttons, inputs, small cards
- `rounded-lg` (8px) — cards, modals, popovers (default for all dialogs)
- `rounded-xl` (12px) — *avoid by default*; reserved for featured marketing modules, not standard cards or modals
- `rounded-full` — circular icon buttons, status dots, pill chips, badges

## 6. Depth & Elevation

PodCodex uses **luminance step**, not box-shadow, for elevation. On the cream canvas, traditional drop shadows look smoke-grey and break the printed-paper feel. On the sepia-black dark canvas, dark-on-dark shadows are invisible anyway.

| Level | Treatment | Use |
|-------|-----------|-----|
| Canvas (0) | `bg-background` | Page background |
| Surface (1) | `bg-card` + `border-border` | Cards, panels |
| Floating (2) | `bg-popover` + `border-border` | Dropdowns, popovers, tooltips |
| Dialog (3) | `bg-popover` + `border-border` + `shadow-lg` over `bg-background/80 backdrop-blur-sm` backdrop | Modals only |

Shadow used only on dialog level — `shadow-lg` is the cap. Never `shadow-2xl`. Tooltips/dropdowns rely on the popover background + border alone. Decorative `shadow-lg` on overlay buttons (e.g. play/download hover overlays on cards) is forbidden — use border or background tint instead.

## 7. Do's and Don'ts

### Do
- Default to `text-sm` for UI density; reserve `text-base` for prose
- Use shadcn semantic tokens (`bg-card`, `text-muted-foreground`); the cream/sepia/gold balance only holds when tokens stay consistent
- Reserve `--primary` (gold-leaf) for one CTA per view and active states
- Use `font-mono` for timestamps, IDs, hashes, file paths — not for body
- Apply `font-feature-settings` correctly: Inter gets `cv11, ss01, ss03` (set on body), Fraunces gets `ss01, onum` (set on h1/h2)
- Use `oklch()` when introducing new color tokens — match perceptual luminance steps of the existing scale
- Two-column settings: label left, control right
- Hover with `hover:bg-accent` (warm tint), not with shadow or scale
- Omit defaults and absent values in labels (don't render "Speaker: —" when there's no speaker)

### Don't
- Don't use pure white (`#ffffff`) or pure black (`#000000`) — both fight the warm canvas
- Don't use `uppercase` or `tracking-wider`. Period. Use weight 500–600 if you need emphasis.
- Don't add waveform visualizations to the AudioBar
- Don't introduce a second chromatic accent — gold-leaf is the only chromatic color
- Don't use Fraunces in body text, buttons, captions, or labels
- Don't elevate cards with `shadow-md`/`shadow-lg` on the canvas — use `border` + `bg-card` instead
- Don't apply zebra-striped lists; rows separate by border or hover only
- Don't introduce em dashes liberally — sparingly, where the prose needs the pause
- Don't bypass the token system with literal hex/oklch values in component code
- Don't disable the paper-grain overlay — it's load-bearing for the editorial feel

### Pipeline stage palette
Pipeline stages get distinct hues for quick recognition. Tokens defined in `index.css`, warm-shifted to coexist with the cream + gold-leaf canvas:

| Stage | Token | Utility | Hue |
|-------|-------|---------|-----|
| Transcribed | `--stage-transcribe` | `bg-stage-transcribe`, `text-stage-transcribe` | 240° (blue) |
| Corrected | `--stage-correct` | `bg-stage-correct`, `text-stage-correct` | 305° (purple) |
| Translated | `--stage-translate` | `bg-stage-translate`, `text-stage-translate` | 195° (teal) |
| Synthesized | `--stage-synth` | `bg-stage-synth`, `text-stage-synth` | 50° (orange) |
| Indexed | `--warning` | `bg-warning/15 text-warning` | 82° (amber — reuses semantic token) |

Each stage chip uses the `bg-{stage}/15 text-{stage}` pattern. If new stages are added, extend `index.css` and this table — don't reach for raw Tailwind colors.

### Exception: media scrims
Elements rendered **on top of artwork or video** (episode card overlays, play buttons, image badges) need contrast against arbitrary image content, not against the theme. The token system would flip in dark mode and break legibility on bright images. Allowed only on top of media:
- `bg-black/{60,65}` and `bg-white/{90,95}` for scrims and floating buttons
- `text-white`, `text-black` paired with the scrims
- `bg-gradient-to-t from-black/60 to-transparent` for bottom-fade overlays
- `shadow-lg` on floating media buttons (play/download) — legibility, not decoration

These never appear on theme surfaces. If you find one on a card or panel without an image behind it, replace with tokens.

## 8. Responsive Behavior

### Breakpoints (Tailwind defaults)
| Name | Width | Behavior |
|------|-------|----------|
| `sm` | ≥640px | Two-column starts for some panels |
| `md` | ≥768px | Settings two-column, sidebar visible |
| `lg` | ≥1024px | Standard desktop layout |
| `xl` | ≥1280px | Generous gutters, wider transcript column |

### Touch targets
- Minimum 32×32 hit area for icon buttons (use `p-2` on a `w-4 h-4` icon)
- Pill chips: `px-2.5 py-1` minimum
- Episode rows: `py-2` minimum row height

### Collapsing strategy
- Sidebar collapses to icons-only at `md` and below; popover label on hover
- Settings two-column → single column at `<md`
- Transcript reader: `max-w-3xl` mobile → `max-w-5xl` desktop
- Modals: full-width sheet on mobile, centered card from `md` up

## 9. Agent Prompt Guide

### When asked to add UI
1. Use shadcn/ui primitives from `frontend/src/components/ui/` first; only build custom when the primitive doesn't fit
2. Reach for tokens: `bg-card`, `text-muted-foreground`, `border-border`, `bg-primary`. No raw hex.
3. Default text size: `text-sm`. Default radius: `rounded-md` for inputs/buttons, `rounded-lg` for cards.
4. Density first: lists use `gap-1.5` to `gap-2` between elements, `py-2` row padding.
5. One CTA per view in `bg-primary`. Everything else is `secondary`, `outline`, or `ghost`.

### Quick reference
- Canvas: `bg-background` / Card: `bg-card` / Popover: `bg-popover`
- Body text: `text-foreground` / Caption: `text-muted-foreground`
- Border: `border-border`
- Primary CTA: `bg-primary text-primary-foreground`
- Hover surface: `hover:bg-accent`
- Mono font: `font-mono` (timestamps, hashes, IDs)
- Display font: `font-display` (h1, h2 only)

### Example prompts
- "Add a settings row with a label `Embedding model` left and a select right. Use two-column at `md`. Label `text-sm font-medium`, helper text `text-xs text-muted-foreground` below the label. Select uses the `.input` utility."
- "Build an episode row: `flex gap-2 py-2 hover:bg-accent rounded-md`. Episode number in `text-xs text-muted-foreground w-8 text-right`, title in `text-sm flex-1 truncate`, date and duration in `text-xs text-muted-foreground` right-aligned. No zebra."
- "Status pill for indexed episode: `inline-flex items-center gap-1 rounded-full bg-success/15 text-success text-2xs font-medium px-2 py-0.5`. Lowercase label `indexed`. No uppercase, no tracking-wider."
- "Modal: `bg-popover border border-border rounded-lg shadow-lg` over `bg-background/80 backdrop-blur-sm` backdrop. Title in `font-display text-2xl` (Fraunces). Body in `text-sm`. Footer right-aligned with `outline` cancel and `default` confirm."

### Iteration guide
1. If colors feel cold or harsh, the warm hue (`oklch H ≈ 55–80`) was lost — re-derive from the existing oklch scale, don't add raw hex
2. If headings feel generic, you forgot `font-display` (Fraunces) or `font-feature-settings: 'ss01', 'onum'`
3. If lists feel cluttered, drop a font size (`text-base` → `text-sm`) before adding spacing
4. If a CTA isn't standing out, check that you used `bg-primary` and that no other element on the view is also primary
5. If something looks "AI-generic SaaS," check for: uppercase labels, `tracking-wider`, gradient buttons, drop shadows on cards, waveform in audio player
