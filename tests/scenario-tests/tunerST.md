# Scenario Tests: Pitch Tuner

**What this is:** User-perspective acceptance criteria for the pitch tuner.
Claude reads this file, uses DevScope to verify each scenario, and iterates on
the code until every scenario passes visually.

**How to use:** Point Claude at this file + DevScope. Claude runs the audio sim,
reads the report + screenshots, compares against these scenarios, fixes code, repeats.

---

## Format

Each scenario has:
- **Setup** — what audio to play, what to look at
- **Pass** — what the user should see (the "golden" behavior)
- **Fail** — what broken looks like (so Claude knows what to fix)
- **DevScope check** — specific DevScope command + what to look for in the output

---

## ST-T01: Steady Note Shows Correct Pitch

**Setup:** Play `Anchors/Ahh easy 1.m4a` at speed 1. Watch `#pitch-note-display` and `#pitch-canvas`.

**Pass:**
- Note display shows a real note name (e.g., "A3", "G4") within 500ms of audio starting
- Note stays stable — doesn't flicker between two notes when the singer holds steady
- Needle sits in or near the green zone (center ±15 cents) for sustained notes
- Cents display shows a small number (e.g., "+3 cents (220 Hz)"), not jumping wildly

**Fail:**
- Note display stays on "—" or shows wrong notes
- Note name flickers rapidly between adjacent notes (e.g., A3 → G#3 → A3 → G#3)
- Needle bounces across the full bar despite steady vocal input
- Cents display shows large values (>30) on a clean sustained note

**DevScope check:**
```bash
devscope.py run vocal-health-coach --audio "Ahh easy" --speed 1 --duration auto
```
- `content_stale` on `pitch-note-display` = FAIL (means it never changed from "—")
- `no_oscillation` on `pitch-note-display` with >4 reversals = FAIL (note flickering)
- Burst capture of `#pitch-canvas` should show needle near center, not bouncing edge to edge:
```bash
devscope.py burst vocal-health-coach --element "#pitch-canvas" --sim-audio "Ahh easy" --speed 1 --fps 10 --duration 3
```

---

## ST-T02: Note Transitions Track Within 300ms

**Setup:** Play `Anchors/Ahh relaxed 2 note.m4a` (two-note anchor). Watch `#pitch-note-display`.

**Pass:**
- When the singer moves from note 1 to note 2, the display updates within ~300ms
- The transition is clean — old note disappears, new note appears, no garbage in between
- Needle smoothly slides from old position to new position (not teleporting)
- During the transition, needle may briefly pass through yellow zone — that's fine

**Fail:**
- Note display lags >1 second behind the actual pitch change
- Display shows an intermediate wrong note during transition (e.g., A3 → C4 → B3 when going A3 → B3)
- Needle teleports (jumps instantly) instead of sliding
- Note never changes — stays on the first note even after the singer moves

**DevScope check:**
```bash
devscope.py burst vocal-health-coach --element "#pitch-note-display, #pitch-canvas" --sim-audio "Ahh relaxed 2 note" --speed 1 --fps 10 --duration 5
```
- In the burst contact sheet, look for exactly 2 distinct note values
- Transition should happen within 3 consecutive frames at 10fps (= 300ms)

---

## ST-T03: Silence = Graceful Idle

**Setup:** Play any audio file. After audio ends, observe tuner for 3+ seconds of silence.

**Pass:**
- When singing stops, needle returns to center (0 cents) or goes to a rest state
- Note display either holds the last detected note or shows "—"
- No phantom notes — silence should NOT produce random note detections
- Canvas doesn't freeze mid-draw (should complete its return animation)

**Fail:**
- Needle keeps bouncing on silence (detecting room noise as pitch)
- Random note names appear during silence ("C2", "B7", etc.)
- Canvas freezes with needle stuck at last position indefinitely
- Display shows erratic values from noise floor

**DevScope check:**
```bash
devscope.py run vocal-health-coach --audio "Ahh easy" --speed 1 --duration auto
```
- After UI_IDLE timestamp, check final burst frames — needle should be centered or static
- `no_oscillation` firing AFTER audio ends = FAIL (noise detection)

---

## ST-T04: Vibrato Detection Activates Correctly

**Setup:** Play audio where singer uses vibrato (any full song recording, e.g., `Danny - Chris Young R1.m4a`).

**Pass:**
- Vibrato badge transitions from "VIBRATO" (gray, idle) to "VIBRATO" (indigo, active) during vibrato passages
- Badge activates during actual vibrato, not during normal sustained notes
- Badge deactivates when vibrato stops (returns to gray)
- Transition is smooth (CSS transition, not a hard toggle)

**Fail:**
- Badge never activates (stays gray entire song)
- Badge activates on every note regardless of vibrato
- Badge flickers rapidly on/off during sustained vibrato
- Badge gets stuck in active state after vibrato ends

**DevScope check:**
```bash
devscope.py run vocal-health-coach --audio "Chris Young" --speed 2 --duration auto
```
- `content_stale` on `vibrato-badge` = check dominant value — if 100% one value, it never toggled
- Burst the badge during a vibrato section to see the transition:
```bash
devscope.py burst vocal-health-coach --element "#vibrato-badge" --sim-audio "Chris Young" --speed 2 --fps 10 --duration 5 --sim-delay 5
```

---

## ST-T05: Fast Passages Don't Break the Tuner

**Setup:** Play `Danny - Runnin down a dream R1.m4a` at speed 1 (fast-tempo rock song).

**Pass:**
- Tuner keeps updating throughout — never gets "stuck" on one note
- Note changes are frequent but each displayed note is plausible (real musical notes, not noise)
- Canvas needle moves actively — following the melody, even if imprecise
- No JS errors, no frozen frames, no blank canvas

**Fail:**
- Tuner can't keep up — shows the same note for 5+ seconds during an active melody
- Display shows non-musical values or garbled text
- Canvas freezes or goes blank during fast passages
- Frame drops spike above 100ms (main thread choked by rapid updates)

**DevScope check:**
```bash
devscope.py run vocal-health-coach --audio "Runnin down" --speed 1 --duration auto
```
- `element_static` on tuner elements = FAIL (means tuner stopped updating)
- `frame_rate` severity_score > 80 = FAIL (too many drops during fast passages)
- `canvas_static` on `pitch-canvas` = FAIL (canvas froze)

---

## ST-T06: Needle Smoothness (No Jitter)

**Setup:** Play `Anchors/Ahh easy 1.m4a`. Burst-capture the pitch canvas at 20fps.

**Pass:**
- Needle position changes by small increments between consecutive frames
- No frame shows the needle jumping more than ~30% of the bar width in one step
- Overall motion path looks smooth when viewed as a filmstrip
- Green zone is the primary home position during sustained clean notes

**Fail:**
- Needle teleports between frames (position jumps >30% of bar width)
- Needle oscillates rapidly (bouncing left-right every frame)
- Needle sits at the same pixel for all frames (not responding to pitch micro-variations)

**DevScope check:**
```bash
devscope.py burst vocal-health-coach --element "#pitch-canvas" --sim-audio "Ahh easy" --speed 1 --fps 20 --duration 3
```
- Read the contact sheet — visually inspect needle positions across frames
- Needle should trace a smooth, mostly-centered path

---

## Scenario Test Status Tracker

| ID | Scenario | Status | Last Run | Notes |
|----|----------|--------|----------|-------|
| ST-T01 | Steady note shows correct pitch | PASS | 2026-03-15 | C#3 stable after warmup, -17 cents consistent, needle in green zone. Fixed: EMA alpha 0.25→0.15, YIN window 250→500ms, stability gate 2→3 frames |
| ST-T02 | Note transitions track within 300ms | PASS | 2026-03-15 | Burst @10fps shows smooth transition, needle moves gradually between positions, no teleportation |
| ST-T03 | Silence = graceful idle | PASS | 2026-03-15 | Note→"—", cents→"—", needle decays to center, vibrato badge gray, IDLE zone. Fixed: frontend clears pitch when `active=false` |
| ST-T04 | Vibrato detection activates correctly | PASS | 2026-03-15 | Badge toggles correctly during vibrato passages, deactivates on silence. Fixed: 500ms hold timer prevents flicker (NO_OSCILLATION resolved) |
| ST-T05 | Fast passages don't break the tuner | PASS | 2026-03-15 | Runnin Down a Dream: tuner keeps updating, plausible notes, no stalls/freezes. No element_static or canvas_static |
| ST-T06 | Needle smoothness (no jitter) | PASS | 2026-03-15 | Burst @20fps: small increments between frames, no teleportation >30% bar width, smooth motion path |
