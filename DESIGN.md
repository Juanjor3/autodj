# AutoDJ Studio — Design System

> A dark, high-contrast, neon-lit electronic audio workstation designed for club DJs and electronic music lovers. Rooted in late-night dance culture (CuBe, The Natural, Metro, Radical), the visual interface combines deep slate-black surfaces, glowing phosphor accents (electric cyan & neon purple), razor-thin 1px borders, and ultra-readable monospace metrics for beatgrids, camelot keys, and real-time transition HUDs.

**Theme:** dark

**Standard:** [RICOUI DESIGN.md](https://github.com/ricocc/ricoui-design-md)

## Tokens — Colors

| Name | Value | Token | Role |
|---|---|---|---|
| canvas | `#070a13` | `--color-canvas` | Deep background canvas behind all panels |
| surface | `#0f172a` | `--color-surface` | Primary card and workspace surface |
| surface card | `#131d35` | `--color-surface-card` | Elevated card container with glassmorphism |
| surface elevated | `#1e293b` | `--color-surface-elevated` | Modal overlays, dropdowns, and flyout controls |
| ink | `#f8fafc` | `--color-ink` | Highest contrast text (headings, active titles) |
| ink muted | `#94a3b8` | `--color-ink-muted` | Secondary body text and descriptive subtitles |
| ink subtle | `#64748b` | `--color-ink-subtle` | Tertiary timestamps, inactive cues, and placeholders |
| hairline | `#1e293b` | `--color-hairline` | Subtle divider lines and structural borders |
| hairline soft | `rgba(255, 255, 255, 0.07)` | `--color-hairline-soft` | Soft glass card perimeter border |
| hairline strong | `rgba(139, 92, 246, 0.35)` | `--color-hairline-strong` | Active accent highlight outline |
| primary purple | `#8b5cf6` | `--color-primary-purple` | Brand signature, active play state, and AI badges |
| accent cyan | `#06b6d4` | `--color-accent-cyan` | Energy meter, transitions, and audio waveform glows |
| accent emerald | `#10b981` | `--color-accent-emerald` | Success states, grid lock, and phase synchronization |
| accent rose | `#f43f5e` | `--color-accent-rose` | Vocal detection, drops, and stop/delete warnings |
| accent amber | `#f59e0b` | `--color-accent-amber` | Ducking status, warnings, and cue markers |

## Tokens — Typography

### Type Scale

| Role | Size | Line Height | Letter Spacing | Token |
|---|---|---|---|---|
| Display | 32px | 1.2 | -0.025em | `--font-size-display` |
| Title Large | 24px | 1.3 | -0.02em | `--font-size-title-lg` |
| Title Medium | 18px | 1.4 | -0.015em | `--font-size-title-md` |
| Body | 14px | 1.5 | 0 | `--font-size-body` |
| Caption | 12px | 1.4 | +0.01em | `--font-size-caption` |
| Micro | 10px | 1.3 | +0.025em | `--font-size-micro` |
| Monospace HUD | 13px | 1.4 | 0 | `--font-size-mono` |

### Font Families
- **UI Primary:** `Inter`, `-apple-system`, `BlinkMacSystemFont`, `Segoe UI`, `sans-serif`
- **Telemetry & Audio Monospace:** `ui-monospace`, `SFMono-Regular`, `Menlo`, `Monaco`, `Consolas`, `monospace`

## Tokens — Spacing & Elevation

| Name | Value | Token | Role |
|---|---|---|---|
| space-xs | 4px | `--space-xs` | Tight inline padding |
| space-sm | 8px | `--space-sm` | Control interior spacing |
| space-md | 16px | `--space-md` | Card interior gap |
| space-lg | 24px | `--space-lg` | Section separator |
| space-xl | 32px | `--space-xl` | Module margins |
| radius-sm | 6px | `--radius-sm` | Badges and small chips |
| radius-md | 12px | `--radius-md` | Buttons, inputs, and track tiles |
| radius-lg | 18px | `--radius-lg` | Panels and dashboard cards |
| glow-neon | `0 0 20px rgba(139, 92, 246, 0.25)` | `--glow-neon` | Active DJ workstation accent glow |

## Components

### Action Button (Primary)
- **Role:** Main trigger for mix generation and download actions.
- **Visuals:** High-contrast gradient or solid purple (`#8b5cf6`) with white text, smooth transition, neon hover glow (`0 0 16px rgba(139, 92, 246, 0.4)`), and slight scale press feedback.
- **States:** Default, Hover, Active, Disabled, Loading (with spinning icon and progress text).

### Audio Track Card
- **Role:** Displays uploaded track metadata, cue points, camelot key wheel, BPM badge, and waveform preview.
- **Visuals:** Dark surface card (`#131d35`) with glass border (`rgba(255, 255, 255, 0.07)`), draggable handle, and responsive action controls.

### Status Badge / Indicator Chip
- **Role:** Shows model connection state (`Modelo Conectado`), stem processing status, and real-time phase alignment.
- **Visuals:** Rounded pill (`px-2.5 py-0.5 rounded-full`) with micro-indicator dot (`animate-pulse` when working, solid emerald when locked).

### Real-Time Telemetry HUD
- **Role:** Byte-by-byte upload tracker, time remaining, instantaneous network transfer rate, and transition countdown.
- **Visuals:** Monospace font metrics, dual-colored gradient progress bar (`from-purple-500 to-cyan-400`), smooth CSS transitions.

## Implementation Guidelines
- **Tailwind CSS Utility Mapping:** Use semantic class variables (`bg-slate-950`, `text-slate-100`, `border-slate-800`, `text-purple-400`, `text-cyan-400`).
- **Dark Mode Continuity:** Never break tonal continuity; no unstyled white flashes.
- **Responsiveness:** Fluid grid from mobile screens up to multi-monitor audio workstation viewports.
