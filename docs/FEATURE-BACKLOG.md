# Koda Vocal Coach — Feature Backlog
# Last updated: March 14, 2026

## Competition deadline: March 16, 2026 @ 5:00 PM PDT

---

## IN PROGRESS / NEXT

### Vocal Range Mapper (MVP in progress)
Real-time piano-roll style visualization. As the singer sings, each note gets colored
green/yellow/red based on average strain at that pitch. Shows passaggio location visually.
- Backend: per-note strain accumulation → `range_update` broadcast events
- Frontend: canvas visualization, current note highlighted, builds live
- No existing consumer app shows this. Strong demo visual.

---

## BUILD NEXT (priority order)

### 1. Audio Streaming to Gemini (HIGH — 40% of judging criteria)
Currently we send Gemini a text description: "Zone: RED, strain: 0.47".
Instead: send the raw PCM audio of the phrase TO Gemini Live, then the strain timeline.
Gemini literally hears the singer and can say "I heard your voice tighten on that last note."
- Resample phrase audio buffer 44.1kHz → 16kHz (Gemini Live requirement)
- Send as inline_data audio part + strain timeline text part
- Changes product from "measurement tool with a voice" to "AI that actually listens"
- Directly hits "Innovation & Multimodal UX" (40% of Gemini judging criteria)
- Est: ~2 hours

### 2. Feedback Ladder ✅ COMPLETE
Tier 1 (silent): green phrases → zone meter only, total silence
Tier 2 (visual): 1st + 2nd consecutive yellow → amber text cue in coach bar, no audio
Tier 3 (voice): RED or 3+ consecutive yellows or breathy → Koda speaks
Constants: YELLOW_VOICE_THRESHOLD = 3
Visual cue text pulled from two levels (mild yellow vs. near-red yellow).
Log output shows: "VOICE / VISUAL / silent" per phrase for easy tuning.
UI: visual_cue event shows amber text in coach bar (no audio).

### 3. Google Cloud Run Deployment (REQUIRED for submission)
Current: running locally / Fly.io. Judges require proof of Google Cloud hosting.
- Containerize with existing Dockerfile
- Deploy to Cloud Run
- Est: ~1 hour

### 4. Demo Video (REQUIRED)
Scripted arc:
  1. Singer sings easy → all green → total silence → "Koda is watching but not interrupting"
  2. Push into passaggio → yellow → visual text cue appears, no voice yet
  3. AC/DC → red → Koda speaks: responds to what it actually heard in the audio
  4. Finish clean song → 4s silence → Koda: "you nailed that"
  5. Show range mapper building in real-time throughout
- Est: ~1 hour

### 5. Architecture Diagram (REQUIRED for submission)
Show: browser mic → WebSocket → EARS analysis → WebSocket → browser
     phrase audio → Gemini Live (audio input) → coaching audio → browser
     Google Cloud hosting layer
- Est: ~30 min

---

## STRETCH GOALS (if time allows)

### Live Warmup Guide
Koda leads an adaptive warmup routine using Gemini Live bidirectional audio.
- "Sing an 'ah' on D4" → EARS detects D4, strain=green → "Good, now try F4..."
- EARS strain + pitch feed back as context each round
- Koda adjusts difficulty based on what it actually hears
- Needs: warmup exercise protocol, bidirectional Gemini Live session management
- Est: ~2-3 hours

### Amazon Nova Parallel Submission (same March 16 deadline — $40K prize pool)
Swap coaching voice to Nova 2 Sonic (Amazon's real-time voice AI).
Same architecture, different AI backbone. Voice AI category fits perfectly.
Submit same app to both competitions.
- Judging: 60% technical, 20% impact, 20% creativity
- Est: ~1-2 hours (after Gemini version is complete)

### Session Persistence / Vocal Progress Tracking
Save range maps across sessions. Show improvement over time.
"Your passaggio has shifted up a semitone since last Tuesday."
- localStorage for MVP, backend DB for production
- Est: ~2 hours

### Lyrics Upload + Display
Upload song lyrics → display with scrolling sync during singing.
Nobody has built lyrics + real-time strain in one place (confirmed gap in research).
- Complex: requires timing sync, file parsing, scrolling display
- Est: ~4-6 hours done right
- Post-competition feature

---

## COMPLETED ✅

- EARS v11 multi-signal strain formula (shimmer_dev + cpp_dev + v8_strain)
- Three-zone calibration (GREEN/YELLOW/RED) confirmed on Daniel's voice (MacBook + browser mic)
- Real-time pitch display (YIN + EMA smoothing + note stability gate)
- Phrase detection with silence hangover (no flicker)
- Gemini Live coaching voice (fires on yellow/red phrases)
- Song-end praise (4s silence + green session → graded praise tier)
- Interruptible audio (stops when phrase_start fires)
- coaching_responses.yaml config (edit wording without code changes)
- live_capture.py (mic → WebSocket → JSONL frame log + WAV)
- playback.py (stream any audio file through backend, hear through speakers)
- Watch Only mode (browser connects without mic for playback testing)
- Reconnect-on-demand for Gemini Live sessions (15-min timeout fix)
- Feedback ladder (silent/visual/voice tiers — YELLOW_VOICE_THRESHOLD=3)
- Vocal range mapper (real-time piano-roll canvas, per-note strain coloring)
- Watch Only mode + playback.py speaker audio + q-to-quit + ping_interval fix
