# UI Action Plan — Production-Grade Polish Sprint

**Goal**: Transform the current functional prototype into a competition-winning, production-grade UI that makes judges want to use our demo video for Gemini marketing materials.

**Deadline**: March 16, 2026 5:00 PM PDT
**Design principle**: "Would a Google DevRel person show this on stage at Cloud NEXT?"

---

## Current State Assessment

The current UI is a **functional prototype** — dark theme, two-panel grid, all features working. But it looks like a developer dashboard, not a consumer product. Specific issues:

1. **Too much data visible** — tonos, thymos, HNR, shimmer bars are debug info that confuses users
2. **Coaching area is tiny** — Gemini's voice coaching (40% of the judging score) is a single line at the bottom
3. **No visual hierarchy** — strain meter and pitch tuner compete for attention equally
4. **No branding** — looks like every dark-mode dashboard ever made
5. **Start overlay is plain** — first impression matters, this is generic
6. **No animation/delight** — everything is static, no micro-interactions
7. **Range map is cryptic** — small colored bars at the bottom with no context

---

## Target UI Architecture

### Layout (top to bottom, single column for focus)

```
┌─────────────────────────────────────────┐
│  HEADER: Logo + Status + Mic Control    │
├─────────────────────────────────────────┤
│                                         │
│     ╭─────────────────────────╮         │
│     │    STRAIN METER (hero)  │         │
│     │   Large arc gauge       │         │
│     │   Score + Zone badge    │         │
│     ╰─────────────────────────╯         │
│                                         │
│  ┌──────────────┐  ┌──────────────┐     │
│  │  PITCH TUNER │  │ VOCAL RANGE  │     │
│  │  Note + Needle│  │ Heat Map     │     │
│  │  Vibrato     │  │              │     │
│  └──────────────┘  └──────────────┘     │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │  SESSION STRAIN GRAPH            │   │
│  │  Timeline with zone coloring     │   │
│  └──────────────────────────────────┘   │
│                                         │
│  ╔══════════════════════════════════╗   │
│  ║  KODA COACHING PANEL (prominent) ║   │
│  ║  Transcript + speaking indicator ║   │
│  ║  Waveform animation when talking ║   │
│  ╚══════════════════════════════════╝   │
└─────────────────────────────────────────┘
```

### Key Design Decisions

1. **Strain meter is the HERO** — largest element, center stage. This is the "money shot" for the demo video. Arc gauge with smooth gradient animation.

2. **Koda coaching panel is PROMINENT** — this is where Gemini lives. When Koda speaks, it should feel alive. Pulsing indicator, transcript appearing word-by-word, subtle waveform.

3. **Debug dimensions HIDDEN by default** — tonos/thymos/HNR/shimmer move to an expandable "Details" section or only show on /debug page. The main UI is clean.

4. **Single column layout** — no split grid. Mobile-friendly by default. Strain meter at top, pitch/range in a compact row below, coaching panel prominent at bottom.

5. **Micro-animations everywhere** — needle slides, gauge sweeps, zone badge pulses on change, coaching text fades in. Nothing snaps — everything transitions.

---

## Phase 1: Layout & Hierarchy Restructure

### Changes:
- Remove debug dimension bars (tonos/thymos/HNR/shimmer) from main UI
- Make strain meter full-width hero section with larger arc gauge (~240px)
- Move pitch tuner + range map into a compact two-column row
- Expand coaching panel to be a proper card with:
  - "Koda" label with speaking indicator (pulsing dot)
  - Transcript text area (larger, readable)
  - Audio playback waveform visualization (subtle)
- Session graph below pitch/range row

### Verification:
- DevScope snap of full page → Gemini prompt: "Rate this UI on a 1-10 scale for production readiness, visual hierarchy, and whether it looks like a hackathon prototype or a real product. Be critical."
- Screenshot comparison: before vs after

## Phase 2: Visual Polish & Branding

### Changes:
- **Color palette refinement**: Current green/yellow/red are fine. Add a signature accent color (indigo/purple for Koda's presence)
- **Typography**: Larger note display (3rem+), thinner weight for secondary info
- **Glow effects**: Strain meter arc should glow in its zone color (subtle bloom)
- **Start overlay redesign**: Large centered mic icon, tagline "Real-time vocal health monitoring", clean CTA button
- **Branded elements**: "Powered by Gemini" badge (subtle, judges will notice)
- **Gradient backgrounds**: Subtle dark gradient panels instead of flat #0d0f14

### Verification:
- DevScope snap → Gemini: "Compare this to Apple Health, Spotify, or Peloton UIs. Does it feel consumer-grade?"
- Watch UI script for live animation check

## Phase 3: Animation & Micro-Interactions

### Changes:
- **Strain gauge sweep animation**: When score changes, arc animates smoothly (CSS transition on stroke-dashoffset or canvas interpolation)
- **Zone badge pulse**: When zone changes (green→yellow), badge briefly scales up 1.05x and glows
- **Pitch needle inertia**: Already has EMA smoothing, verify it looks fluid in burst capture
- **Coaching panel entrance**: When Gemini speaks, panel slides up or text fades in character by character
- **Vibrato badge shimmer**: When active, subtle shimmer/pulse animation on the indigo background
- **Session graph live drawing**: New data points animate in from the right edge

### Verification:
- DevScope burst capture of strain gauge during zone transition
- DevScope burst of coaching panel during Gemini speech event
- Full DevScope run → check no frame rate regression from animations

## Phase 4: Gemini Integration Visibility

### THIS IS CRITICAL FOR SCORING (40% Innovation + 30% Demo weight)

### Changes:
- **"Listening" state**: When voice is active, show a subtle waveform/amplitude bar near the coaching panel — shows Gemini is perceiving
- **Coaching delivery**: When Gemini speaks:
  - Panel background subtly lights up (indigo tint)
  - Transcript appears with a typing animation or word-by-word reveal
  - Small speaker icon pulses while audio plays
  - After audio finishes, transcript stays for 8s then fades
- **Coaching history**: Last 2-3 coaching messages visible as faded cards below current one
- **"Powered by Gemini Live" badge**: Small, tasteful, bottom corner

### Verification:
- Watch UI with live mic → confirm coaching flow looks natural
- DevScope run → verify coaching elements update without content_stale

---

## Feedback Loop Tools

### 1. DevScope (automated, headless)
```bash
# Full analysis run
devscope.py run vocal-health-coach --audio "Chris Young" --speed 2 --duration auto

# Element-specific burst for animation smoothness
devscope.py burst vocal-health-coach --element "#strain-canvas" --sim-audio "Ahh easy" --speed 1 --fps 20 --duration 3

# Quick snapshot for layout check
devscope.py snap vocal-health-coach --label "after-phase1"
```

### 2. Watch UI (Playwright screenshots during live sim)
```bash
python scripts/watch_ui.py --url http://localhost:8765/test --speed 2 --interval 1 --duration 20
# Outputs to /tmp/koda_watch/shot_*.png — read with Read tool
```

### 3. Gemini Vision Feedback (for subjective UI quality)
After each phase, take a screenshot and ask Gemini 2.0 Flash:

**Prompt template:**
```
You are judging a hackathon entry for the Gemini Live Agent Challenge.
This app is a real-time vocal health coach — it detects vocal strain while
someone sings and provides coaching through Gemini Live voice API.

Scoring criteria:
- Innovation & Multimodal UX (40%): Does it break the "text box" paradigm?
  Does the UI feel alive and responsive to voice?
- Technical Implementation (30%): Does the UI feel production-grade?
  Clean architecture visible?
- Demo & Presentation (30%): Would you remember this demo?
  Does it look polished enough to show at Google Cloud NEXT?

Rate this screenshot 1-10 on:
1. Production readiness (would you ship this?)
2. Visual hierarchy (is the most important thing the most visible?)
3. "Wow factor" (would a judge pause and look closer?)
4. Gemini visibility (can you tell Gemini is involved just by looking?)

Be brutally honest. What's the #1 thing to improve?
```

### 4. Safari Live Preview
Open http://localhost:8765 in Safari. Sing. Take CMD+Shift+3 screenshots.
Compare before/after each phase.

---

## Timeline (remaining ~24 hours)

| Time | Task | Duration |
|------|------|----------|
| Now | Phase 1: Layout restructure | 2-3 hrs |
| After Phase 1 | DevScope + Gemini feedback | 15 min |
| Evening | Phase 2: Visual polish + branding | 1-2 hrs |
| Evening | Phase 3: Animations | 1 hr |
| Evening | Phase 4: Gemini integration visibility | 1 hr |
| Late evening | GCP deploy (deploy-gcp.sh) | 15 min |
| Late evening | Demo video recording | 1 hr |
| Night/morning | Blog post (dev.to) + GDG signup | 30 min |
| Morning | Final submission on Devpost | 30 min |

---

## Non-Negotiables

1. **All existing functionality MUST keep working** — strain detection, pitch tuner, vibrato, coaching, range map
2. **WebSocket contract unchanged** — frontend changes only, no backend modifications needed
3. **DevScope scenario tests still pass** — run ST-T01 through ST-T06 after UI changes
4. **Mobile-responsive** — judges might open on phone
5. **Performance** — no new frame rate regressions from animations
