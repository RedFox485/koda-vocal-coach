# Koda Vocal Health Coach — Demo Video Production Bible

**Competition:** Gemini Live Agent Challenge
**Deadline:** March 16, 2026, 6:00 PM MDT
**Target length:** 2:30–2:45 (hard max: 4:00)
**Upload to:** YouTube (public or unlisted)

---

## Strategic Framing

**Core thesis:** Koda is a real-time vocal health coach that uses multiple parallel acoustic analyzers and a persistent Gemini Live session to detect strain and speak technique cues at each breath point — giving every singer in the world access to feedback that used to require a vocal coach.

**Decisions (March 16 AM):**
- Closing line: "Most people who love to sing will never work with a vocal coach. Koda changes that. Powered by Gemini Live."
- Screen recording: Reel (programmatic Playwright capture, smooth 30fps)
- Singing voice: CC0 Courtney Odom acapella (injected via Reel, best strain dynamics)
- Architecture diagram: Dark theme (already rendered at docs/architecture-LATEST.png)
- Video editing: CutRoom (local FFmpeg render, no GPU needed)
- Blog post: dev.to technical writeup + DevPost submission text
- Voiceover: All 9 clips generated, vo_09 regenerated with new closing line

**Why this wins:**
- The demo is inherently cinematic — singing + AI voice coaching in real-time
- DJI opening shot immediately signals "this is real, not a mockup"
- Named algorithms (Parselmouth, CPP, perceptual engine) prove depth beyond a wrapper
- The access/impact closer gives judges a "why" that resonates
- Gemini Live is the hero: persistent session, native audio generation, breath-point timing

**Emotional arc:** Curiosity (hook) → Understanding (how it works) → Awe (Gemini coaching in real-time) → Purpose (why it matters)

---

## Voice Direction

**Tool:** ElevenLabs (free tier: 10 min/month). Fallback: Microsoft Edge TTS (free, neural voices).

**Voice character:** Male, 30s. Calm, grounded authority. Think: senior engineer giving a conference demo they're genuinely proud of. NOT a product commercial. NOT a hype reel. The confidence comes from the tech being real, not from the narrator selling it.

**Pacing:** ~140 words/min (slightly slower than conversational). Deliberate pauses after key technical terms — let them land. When naming algorithms, slight emphasis but not dramatic.

**Emotional modulation by shot:**
| Shot | Energy | Tone |
|------|--------|------|
| 2 (Problem) | Low, grounded | Matter-of-fact. "This is the reality." |
| 3 (App intro) | Neutral, clean | Simple. Let Gemini's voice be the star. |
| 4 (Green zone) | Steady, technical | Engineering confidence. Naming the stack. |
| 5 (Yellow/red) | Slightly elevated | Engaged. "Watch this." Energy rises with the strain. |
| 5b (Coaching cue) | Drops to quiet | Step back. Let Gemini speak. Then explain, impressed. |
| 6 (Summary) | Warm | Genuine appreciation for what just happened. |
| 7 (Architecture) | Crisp, fast | Rapid-fire technical. Showing command of the stack. |
| 8 (Close) | Slow, resonant | The "why." Let it breathe. |

**ElevenLabs settings (if using):**
- Stability: 0.65 (slight natural variation, not robotic)
- Clarity: 0.75
- Style exaggeration: 0.3 (subtle warmth)

---

## Shot-by-Shot Production Plan

**Narration math:** At 140 wpm, 10 seconds = ~23 words. Every word is budgeted below.

---

### Shot 1: Hook (0:00–0:03) — 3 seconds

**Type:** DJI Osmo 2, real camera
**Visual:** Close-up of Daniel singing into his phone. Koda UI visible on the phone screen — strain gauge moving. Warm lighting.
**Audio:** Natural room sound — Daniel's voice, maybe a faint hint of Koda's coaching from the phone speaker. Raw and real.
**Voiceover:** None.
**Text overlay:** None.

**Why 3 seconds:** Hooks need to be instant. DJI footage is the "pattern interrupt" — judges expect screen recordings, they get a real human singing. 3 seconds is long enough to register, short enough to leave them wanting more.

**Pass criteria:**
- [ ] Daniel is visibly singing (mouth open, engaged expression)
- [ ] Phone screen shows Koda UI with strain gauge visible (even if small)
- [ ] Camera is stable (DJI Osmo stabilization)
- [ ] Audio is natural — not silent, not clipping
- [ ] Cut to Shot 2 is clean (hard cut, no transition)

---

### Shot 2: Problem + Solution (0:03–0:12) — 9 seconds

**Type:** Screen recording — Koda start screen (idle state, before clicking Start)
**Visual:** Clean Koda UI. Start overlay visible.

**Voiceover (22 words, ~9s at 140 wpm):**
> "Hundreds of millions of people sing without a teacher. Nobody tells them when they're hurting their voice. This is Koda."

**Text overlay:** None. The words carry it. The UI sells itself.

**Why 9 seconds (not 12 from before):** The old version had two separate concepts — problem statement AND solution overview. Merging them into one sentence is tighter. "This is Koda" is the bridge — it names the product and implies "here's the fix" without wasting a sentence explaining it. Judges are smart; they'll get it from what follows.

**Voice direction:** Low energy. Matter-of-fact. "Hundreds of millions" should feel weighty but not dramatic. "This is Koda" is simple and clean — slight pause before it, like you're introducing someone important.

**Pass criteria:**
- [ ] Koda UI is clean, centered, visually appealing
- [ ] Start overlay is visible (shows this is the beginning of a session)
- [ ] No cursor jitter or window chrome visible
- [ ] Voiceover timing: "This is Koda" lands as the eye settles on the UI

---

### Shot 3: Gemini Greeting (0:12–0:20) — 8 seconds

**Type:** Screen recording — click Start, Gemini speaks greeting
**Visual:** Cursor clicks Start → overlay dismisses → Koda panel lights up → Gemini's greeting text streams in the coaching panel.

**Voiceover (5 words, ~2s):**
> "One tap. Gemini introduces itself."

**Audio:** Let Gemini's spoken greeting play (~4-5s). This is the first time judges HEAR Gemini's voice. Don't talk over it.

**Sequence:**
1. 0:12 — Voiceover: "One tap. Gemini introduces itself." (2s)
2. 0:14 — Click Start. Overlay dismisses.
3. 0:15 — Gemini greeting plays in full (~5s): "Ready when YOU are!" or similar
4. 0:20 — Cut to singing

**Why 8 seconds:** Judges need to hear Gemini immediately — it's 40% of the score (Innovation & Multimodal UX). But we don't need to explain what's happening. The greeting is self-evident. Keep it tight.

**Voice direction:** Quick and clean. "One tap" is almost throwaway — the star of this shot is Gemini's voice, not the narrator's.

**Pass criteria:**
- [ ] Click-to-greeting latency < 3 seconds (no awkward dead air)
- [ ] Gemini's voice is clearly audible and natural-sounding
- [ ] Coaching panel visually reacts (text streams in, panel lights up)
- [ ] Voiceover does NOT overlap with Gemini's greeting
- [ ] Transition to singing feels natural (not abrupt)

---

### Shot 4: Green Zone — Technical Showcase (0:20–0:48) — 28 seconds

**Type:** Screen recording — full Koda UI, Daniel singing easy/comfortable passage
**Visual:** Strain gauge solidly green. Pitch tuner tracking. Range map building out. Session graph starting to populate.

**Audio:** Daniel singing (full volume in mix for first ~5 seconds, then duck under voiceover to ~40% volume). Voiceover enters at ~0:25.

**Voiceover (52 words, ~22s at 140 wpm):**
> "Multiple acoustic analyzers run in parallel on every audio frame. Parselmouth extracts shimmer and harmonic-to-noise ratio. A cepstral peak prominence detector measures vocal fold closure. An eight-channel perceptual engine scores timbral strain. Total pipeline latency: ninety-six milliseconds. Green means healthy phonation — the baseline adapts to your voice, not fixed thresholds."

**Text overlays (subtle, lower-third, appear/fade as mentioned):**
- "Parselmouth · Shimmer · HNR" (when mentioned, ~2s)
- "CPP · Vocal Fold Closure" (when mentioned, ~2s)
- "8-Channel Perceptual Engine" (when mentioned, ~2s)
- "96ms Pipeline Latency" (stays for ~3s, slightly larger)

**Sequence:**
1. 0:20–0:25 — Daniel sings. No voiceover. Let judges hear the singing and see the gauge responding. This is "proof of life."
2. 0:25–0:47 — Voiceover explains the technical stack while Daniel continues singing underneath. Gauge stays green throughout.
3. 0:47–0:48 — Brief pause. Singing continues. Transition to harder passage.

**Why 28 seconds:** This is where we prove technical depth. Google DevRel judges want to know you built something real. Naming Parselmouth, CPP, and the perceptual engine by name signals "I understand the signal processing, this isn't a wrapper." The green zone also establishes the baseline visually — so when it shifts to yellow, the contrast is obvious.

**Voice direction:** Steady engineering confidence. Each analyzer name gets slight emphasis. "Ninety-six milliseconds" should land with understated pride — like casually mentioning a fast lap time. "Not fixed thresholds" is the technical differentiator — slight stress on "your voice."

**Pass criteria:**
- [ ] Strain gauge is SOLIDLY green for entire shot (no yellow flickers)
- [ ] Pitch tuner is actively tracking (proves real-time audio processing)
- [ ] Range map shows at least some dots populating
- [ ] Text overlays appear/disappear cleanly (not distracting)
- [ ] Daniel's singing is audible under voiceover (proves this is live, not a mockup)
- [ ] No gauge lag visible — gauge moves with the singing

---

### Shot 5: Yellow/Red Zone — The Money Shot (0:48–1:38) — 50 seconds

This is the most important shot in the entire demo. It contains three sub-moments:

#### 5a: Strain Builds (0:48–1:05) — 17 seconds

**Visual:** Daniel pushes into chorus/belt. Gauge shifts from green → yellow. Maybe touches red briefly.
**Audio:** Daniel singing harder (full volume, visceral). Voiceover enters after ~5 seconds.

**Voiceover (30 words, ~13s):**
> "When technique breaks down, Koda sees it immediately. Shimmer spikes as vocal fold vibration becomes irregular. Cepstral peak prominence drops. The strain engine fuses these signals in real-time."

**Voice direction:** Energy rises with the strain. Narrator is engaged — "watch this" energy. "Immediately" and "real-time" get slight stress.

#### 5b: Coaching Cue — THE Critical Moment (1:05–1:18) — 13 seconds

**Visual:** Daniel pauses to breathe. Gemini coaching cue appears in panel and plays through speakers.
**Audio:** Singing stops → brief silence (1s) → Gemini speaks coaching cue (~4-5s, e.g. "Ease off the push — find the note with less effort, not more.") → brief silence (1s)

**Voiceover (NONE during the cue).** Let Gemini speak uninterrupted. This is the moment judges will remember.

**After cue plays, voiceover (26 words, ~11s):**
> "At every breath point, Gemini speaks a technique cue. Not scripted — it reads the strain score, phrase duration, and vocal mode, then improvises."

**Text overlay:** "Phrase Coaching — Gemini Live" (subtle lower-third, appears when cue starts, fades after voiceover)

#### 5c: Second Phrase + Second Cue (1:18–1:38) — 20 seconds

**Visual:** Daniel sings another hard phrase → pauses → second coaching cue from Gemini.
**Audio:** Singing → pause → Gemini speaks again (different cue this time, proving it's not pre-recorded).

**Voiceover (22 words, ~10s, AFTER second cue):**
> "Every cue is different. Gemini improvises within technique-anchored style guidelines, using the last five phrases of context."

**Why two coaching cues:** One cue could be scripted. Two different cues prove Gemini is actually reading the session and improvising. This is the "Beyond Text" factor that's 40% of the score.

**Voice direction for 5b/5c:** Drop to quiet reverence when Gemini speaks. After the cue, the narrator sounds genuinely impressed — "this just happened, and it was different from the last one." Not fake surprise, but real appreciation.

**Pass criteria for entire Shot 5:**
- [ ] Gauge visibly shifts from green to yellow (or red) — the contrast with Shot 4 must be obvious
- [ ] FIRST coaching cue: Gemini speaks clearly, no audio artifacts, relevant to what just happened
- [ ] SECOND coaching cue: DIFFERENT from the first (proves improvisation, not playback)
- [ ] No voiceover overlaps with Gemini's voice — ever
- [ ] Coaching panel text streams in during both cues
- [ ] Transition from singing → pause → cue feels natural (not awkward silence)
- [ ] Daniel's singing in the harder passage sounds genuinely pushed (not faking strain)

---

### Shot 6: Song End — Gemini Summary (1:38–2:05) — 27 seconds

**Type:** Screen recording
**Visual:** Daniel stops singing. Session graph shows the full green/yellow/red pattern of the song. After ~4 seconds of silence, Gemini speaks the song-end summary.

**Sequence:**
1. 1:38–1:42 — Daniel stops. Silence. Gauge settles. (4s)
2. 1:42–1:52 — Gemini speaks song-end summary (~10s). Let it play in FULL. (e.g., "Really strong session — mostly green with just a little push at the edges. You found it and held it.")
3. 1:52–2:05 — Voiceover explains.

**Voiceover (30 words, ~13s):**
> "After each song, Gemini delivers a spoken summary of the entire session. One persistent connection — every phrase, every strain spike, every zone transition stays in context. No resets."

**Text overlay:** "Song Summary — Full Session Context" (lower-third during summary playback)

**Voice direction:** Warm. The summary is a satisfying conclusion to the singing — the narrator mirrors that energy. "No resets" is the technical kicker — land it cleanly.

**Pass criteria:**
- [ ] 4-second silence before summary triggers naturally (not edited/forced)
- [ ] Gemini's summary references specifics from the actual session (proves context retention)
- [ ] Summary audio is clean and fully audible
- [ ] Session graph on screen shows visible green/yellow/red patterns from the singing
- [ ] Voiceover does NOT overlap with Gemini's summary

---

### Shot 7: Architecture Flash (2:05–2:20) — 15 seconds

**Type:** Architecture diagram PNG (our rendered diagram), fullscreen
**Visual:** Clean architecture diagram fills the screen. Maybe a subtle slow zoom (105% over 15s).

**Voiceover (38 words, ~16s — slightly faster pace here, ~150 wpm, rapid-fire technical):**
> "Browser captures audio over WebSocket at ten hertz. Cloud Run backend runs multiple analyzers in parallel — Parselmouth, CPP, perceptual engine, wavelet scattering, phonation classifier. The strain engine fuses signals and triggers Gemini Live at each phrase boundary. Native audio generation — not TTS."

**Text overlay:** None — the diagram IS the visual. Keep it clean.

**Voice direction:** Crisp. Faster than the rest of the video. This is the "I know exactly how this works" moment. Each component name lands like a checklist item. "Native audio generation — not TTS" is the closer — slight pause before "not TTS," like you're preempting their question.

**Why 15 seconds (not 20):** Oleksandr's rule: technical implementation = 15-30 seconds max. The judges who care about architecture will pause the video on this frame. The judges who don't will appreciate that you didn't dwell. 15 seconds is enough to name every component and show the diagram.

**Pass criteria:**
- [ ] Architecture diagram is sharp and readable at YouTube 1080p
- [ ] Voiceover covers all 5 analyzers by name
- [ ] Pace is noticeably faster than rest of video (signals command)
- [ ] Ends cleanly — no trailing silence on the diagram

---

### Shot 8: Impact Close (2:20–2:40) — 20 seconds

**Type:** Return to Koda UI (screen recording), then clean title card for final 5 seconds.

**Visual sequence:**
1. 2:20–2:32 — Koda UI, relaxed state. Maybe the session graph is visible showing the full practice.
2. 2:32–2:40 — Clean title card: "Koda" logo/text, "Real-time vocal health coaching," "Powered by Gemini Live" badge. Dark background, minimal.

**Voiceover (~8s):**
> "Most people who love to sing will never work with a vocal coach. Koda changes that. Powered by Gemini Live."

**Text overlay on title card:**
- "Koda" (large)
- "Real-time vocal health coaching" (subtitle)
- "Powered by Gemini Live" (badge, bottom corner)

**Voice direction:** This is the emotional peak. Slow down. "Eighty dollars an hour" should feel like a real barrier. "If you can find a teacher at all" widens the lens — this isn't just about cost, it's about access. The pause before "for free" is critical — let the audience arrive at it before the narrator confirms it. "Powered by Gemini Live" is the clean button — said with pride, not hype.

**Why the access angle closes instead of opens:** Opening with stats feels like a pitch deck. Opening with singing feels real. By the time judges reach this moment, they've already seen the tech work, they've heard Gemini coaching, they believe it's real. NOW you tell them why it matters. The emotional payoff is earned, not assumed.

**Pass criteria:**
- [ ] Title card is clean, professional, readable
- [ ] "Powered by Gemini Live" is visible but not dominant
- [ ] Voiceover pace is noticeably slower than Shot 7 (emotional contrast)
- [ ] Final silence is clean — video ends crisply, no awkward trail-off
- [ ] Total runtime: 2:35–2:45

---

## Time Budget Summary

| Shot | Duration | Voiceover Words | Content |
|------|----------|----------------|---------|
| 1. Hook | 3s | 0 | DJI real-world footage |
| 2. Problem + Solution | 9s | 22 | Problem statement + "This is Koda" |
| 3. Gemini Greeting | 8s | 5 | One tap + Gemini speaks |
| 4. Green Zone | 28s | 52 | Technical stack showcase |
| 5a. Strain Builds | 17s | 30 | Gauge shifts, explain signals |
| 5b. Coaching Cue #1 | 13s | 26 | Gemini coaches (let it play!) |
| 5c. Coaching Cue #2 | 20s | 22 | Second cue proves improvisation |
| 6. Song Summary | 27s | 30 | Gemini summarizes full session |
| 7. Architecture | 15s | 38 | Rapid-fire tech diagram |
| 8. Impact Close | 15s | 20 | Access angle + title card |
| **TOTAL** | **~2:40** | **~260** | |

**Total voiceover: ~260 words at ~140 wpm = ~1:51 of narration**
**Remaining ~49 seconds: Gemini's voice, singing, silence, and breathing room**

---

## Full Script (Final — Generate Voiceover From This)

Generate each numbered block as a separate audio file. Silence/Gemini moments are marked [BREAK] — these are NOT narrated.

```
[1] — SHOT 2 (0:03)
Hundreds of millions of people sing without a teacher. Nobody tells them when they're hurting their voice. This is Koda.

[2] — SHOT 3 (0:12)
One tap. Gemini introduces itself.

[BREAK: Gemini greeting plays ~5s]

[3] — SHOT 4 (0:25)
Multiple acoustic analyzers run in parallel on every audio frame. Parselmouth extracts shimmer and harmonic-to-noise ratio. A cepstral peak prominence detector measures vocal fold closure. An eight-channel perceptual engine scores timbral strain. Total pipeline latency: ninety-six milliseconds. Green means healthy phonation — the baseline adapts to your voice, not fixed thresholds.

[4] — SHOT 5a (0:53)
When technique breaks down, Koda sees it immediately. Shimmer spikes as vocal fold vibration becomes irregular. Cepstral peak prominence drops. The strain engine fuses these signals in real-time.

[BREAK: Daniel pauses. Gemini coaching cue #1 plays ~5s]

[5] — SHOT 5b (1:12)
At every breath point, Gemini speaks a technique cue. Not scripted — it reads the strain score, phrase duration, and vocal mode, then improvises.

[BREAK: Daniel sings another phrase. Pauses. Gemini coaching cue #2 plays ~5s]

[6] — SHOT 5c (1:28)
Every cue is different. Gemini improvises within technique-anchored style guidelines, using the last five phrases of context.

[BREAK: Daniel stops singing. 4s silence. Gemini song summary plays ~10s]

[7] — SHOT 6 (1:52)
After each song, Gemini delivers a spoken summary of the entire session. One persistent connection — every phrase, every strain spike, every zone transition stays in context. No resets.

[8] — SHOT 7 (2:05)
Browser captures audio over WebSocket. Cloud Run backend runs multiple analyzers in parallel — Parselmouth, C P P, perceptual engine, phonation classifier. The strain engine triggers Gemini Live at each phrase boundary. Native audio generation.

[9] — SHOT 8 (2:20)
Most people who love to sing will never work with a vocal coach. Koda changes that. Powered by Gemini Live.
```

---

## Production Plan

### Screen Recording Setup — Reel (Programmatic Capture)
- **Tool:** Reel (`~/Documents/projects/reel/reel.py`) — Playwright headless browser capture
- **Mode:** Hybrid (default) — WebSocket injection works naturally, 30fps capture
- **Resolution:** 1920x1080 at device_scale_factor=1 (clean, no DPR issues)
- **Audio:** Injected via `/inject/ws` endpoint — Courtney Odom CC0 vocals
- **Advantages over manual recording:** Frame-perfect, no jitter, repeatable, no browser chrome
- **Captures are automated** — Reel injects audio and captures simultaneously

### DJI Osmo 2 Shot
- Stabilized close-up: Daniel from chest-up, phone visible in hand or on stand
- Koda UI should be visible on the phone screen (even if small — it sells authenticity)
- Good lighting: ring light or window light. Warm tones preferred.
- Record in 1080p. 2-3 seconds of clean footage is all we need.
- Record 30+ seconds — we'll pick the best 3.

### Song Choice
- **Primary:** Liza Jane — proven verse/chorus strain separation in testing
- **Backup:** Any song Daniel knows well with clear easy/hard sections
- **Key:** Comfort > complexity. A confident performance sells better than a showy one.
- **Requirement:** The song MUST produce at least one yellow zone moment and at least two natural breath pauses (for coaching cues)

### Voiceover Generation
1. Split the script above into 9 separate text blocks
2. Generate each as a separate audio file in ElevenLabs
3. Name files: `vo_01_problem.mp3`, `vo_02_greeting.mp3`, etc.
4. Listen to each — re-generate any that sound rushed, robotic, or overly dramatic
5. Target: calm confidence, slight warmth, no hype

### Editing — CutRoom (Programmatic Video Editor)
**Tool:** CutRoom (`~/Documents/projects/cutroom/cutroom.py`) — JSON timeline → FFmpeg render

**Audio layers (4 tracks):**
1. Daniel's singing (from screen recording)
2. Gemini's voice (from screen recording system audio)
3. Voiceover (generated clips)
4. Optional: very subtle ambient pad (only during architecture shot — helps the rapid-fire feel less dry)

**Mixing rules:**
- Singing at 100% volume when no voiceover is playing
- Singing ducks to ~40% when voiceover enters
- Gemini's voice is ALWAYS at 100% — never duck it
- Voiceover and Gemini never overlap — this is non-negotiable
- No background music during singing or coaching moments

**Text overlays:**
- Font: SF Pro or system sans-serif. White text, slight drop shadow.
- Appear: fade in over 0.3s. Disappear: fade out over 0.3s.
- Position: lower-third (bottom 15% of frame)
- Only the overlays listed in each shot above — no extras

**Transitions:**
- Hard cuts only. No dissolves, no wipes, no fancy effects.
- Exception: DJI → screen recording can have a very fast cross-dissolve (0.2s)

---

## Pre-Recording Checklist

- [ ] Koda working locally — test run: sing a song, verify gauge responds, Gemini coaches
- [ ] At least 2 coaching cues fire during test run (verify phrase detection works)
- [ ] Song-end summary fires after 4s silence (verify timer works)
- [ ] DJI Osmo 2 charged + stabilizer calibrated
- [ ] Good lighting setup (test with phone camera first)
- [ ] Quiet room — no AC hum, no traffic, no background noise
- [ ] Screen recording tested — both screen video AND system audio captured
- [ ] Chrome in full-screen, no bookmarks bar, no extensions visible
- [ ] ElevenLabs account ready with test clip generated
- [ ] Koda deployed on Cloud Run (for submission proof — recording can be local)

---

## Shot Evaluation Rubric

After recording, evaluate each shot against these criteria. A shot needs ALL items checked to be a PASS.

| Shot | Must-Have | Nice-to-Have |
|------|-----------|-------------|
| 1 (Hook) | Singing visible, phone screen visible, stable | Koda gauge visible on phone |
| 2 (Problem) | Clean UI, no jitter | — |
| 3 (Greeting) | Gemini speaks < 3s after click, voice clear | Panel animation visible |
| 4 (Green) | Gauge SOLID green, pitch tuner tracking | Range map populating |
| 5a (Strain) | Gauge moves to yellow, visible contrast vs Shot 4 | Touches red briefly |
| 5b (Cue #1) | Gemini cue is clear, relevant, uninterrupted | Coaching panel lights up |
| 5c (Cue #2) | DIFFERENT from cue #1, relevant | Natural pause-to-cue timing |
| 6 (Summary) | Summary fires, references session specifics | Session graph visible |
| 7 (Arch) | Diagram is sharp, readable at 1080p | — |
| 8 (Close) | Title card is clean, "Powered by Gemini Live" visible | Emotional pacing right |

---

## Key Principles

1. **Authenticity > polish.** Real singing, real strain, real Gemini coaching. No mockups.
2. **Show, don't tell.** The product IS the demo. Let Gemini speak.
3. **Technical depth earns respect.** Name the algorithms. Name the latencies. Google judges know the difference between a wrapper and real engineering.
4. **Gemini must be heard.** The coaching cues and song summary are the "wow" moments. NEVER talk over them.
5. **The access angle is earned, not assumed.** Show the tech first, explain the "why" last. Impact hits harder after proof.
6. **No filler.** Every second earns its place. 2:40 is the target.

---

## Timeline (March 16, 2026)

| Task | When | Duration |
|------|------|----------|
| Test recording (singing + Koda) | Morning, early | 30 min |
| DJI Osmo 2 shot | After test | 15 min |
| Screen recording (3+ full takes) | Mid-morning | 45 min |
| Generate voiceover (9 clips) | After recording | 30 min |
| Edit in iMovie/DaVinci | Afternoon | 1.5-2 hours |
| Review + re-edit | After first cut | 30 min |
| Upload to YouTube | By 4:00 PM MDT | 15 min |
| Submit on DevPost | By 5:00 PM MDT | 30 min |
| **Buffer** | 5:00–6:00 PM | Emergency fixes only |

---

## Winning Strategy — What Past Winners Did

### Gemini API Developer Competition 2024 Winners

**Jayu (Best Overall App)** — Personal AI assistant with screen vision
- Built by a UCLA CS student (Jonathan Ouyang)
- Demo showed the assistant ACTUALLY controlling applications on screen in real-time
- Won because it demonstrated genuine multimodal capability — not a chatbot wrapper
- Key: the demo was the product. No slides needed.

**ViddyScribe (Best Web App)** — Auto-generates audio descriptions for YouTube videos
- Clear accessibility problem → clear solution → immediate value
- Demo: upload video → descriptions generated → play back with descriptions
- Won because judges could understand the value in 10 seconds
- Featured by Chrome for Developers team afterward

**Gaze Link (Best Android App)** — ALS communication system using eye tracking
- Deeply personal problem (built for someone with ALS)
- Demo showed a real person communicating through eye movements
- Won on emotional impact + genuine technical innovation
- Key: the "why" was undeniable

### Patterns Across All Winners

1. **Solve a REAL problem for REAL people.** Every winner addressed something specific and human — not a generic "AI assistant." Koda fits this: self-taught singers genuinely need strain feedback.

2. **The demo IS the product.** Winners didn't explain their product — they showed it working. The best demos had zero slides. Screen recording of the app doing its thing, narrated simply.

3. **Multimodal fluidity.** Winners that scored highest on Innovation & Multimodal UX (40%) showed Gemini doing something that couldn't be done with text alone. Koda's voice coaching at breath points is exactly this.

4. **Under 3 minutes.** Most winning demos were 2-2.5 minutes. Judges review dozens back-to-back — concision signals confidence and respect for their time.

5. **Authenticity over production.** Real terminal execution, real user interactions, real results. A polished mockup video loses to a rough screen recording of something genuinely working.

6. **Hook in 10 seconds.** From Oleksandr (won 7 of 15 hackathons): judges decide in 30 seconds whether to care. The DJI opening shot gives us that hook.

### What 7-Time Winner Oleksandr Recommends

Structure (mandatory):
1. Problem definition backed by data (10-15 sec)
2. Solution overview (15-20 sec)
3. Live app demo from USER perspective (60-90 sec)
4. Technical implementation — high level only (15-30 sec)
5. Closing — impact + future (10-15 sec)

### What Judges Hate

- Long intros, company logos, team introductions
- Setup flows (login screens, config, "first let me create an account...")
- Marketing language without substance
- Monotone narration
- Videos that feel exhaustion-produced (rambling at 3 AM the night before)
- Mockups or fake demos

### Our Competitive Advantages

1. **The demo is inherently cinematic.** Someone singing → AI responding with voice coaching in real-time is more compelling than most app demos. Judges will HEAR Gemini coaching.
2. **Real-world DJI shot.** Most entries are pure screen recordings. Opening with a real camera shot of Daniel actually singing into his phone immediately signals "this is different."
3. **The problem is visceral.** Every judge has either sung too hard or knows someone who has. "Your voice is hoarse the next morning" lands instantly.
4. **Gemini Live is the hero.** The persistent session, native voice, breath-point timing — this showcases Gemini's capabilities in a way that most chatbot wrappers can't.
5. **Technical depth is real.** Named algorithms, measured latencies, session-adaptive baselines. This isn't a tutorial project — it's signal processing research with a Gemini integration.
6. **The access angle.** Voice lessons cost $80/hour — most of the world can't access one. This gives judges a "why" that goes beyond "cool tech demo."

### Sources
- [Devpost: 6 Tips for Making a Winning Demo Video](https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video)
- [Devpost: Oleksandr's Secret to Winning 7/15 Hackathons](https://info.devpost.com/blog/user-story-oleksandr)
- [Google Developers Blog: Gemini Competition Winners](https://developers.googleblog.com/en/announcing-the-winners-of-the-gemini-api-developer-competition/)
- [UCLA: Jayu Winner Profile](https://samueli.ucla.edu/ucla-computer-science-student-winner-of-google-ai-contest-dreams-big-of-building-more-intelligent-tools/)
- [Chrome for Developers: ViddyScribe](https://developer.chrome.com/blog/video-accessibility-gemini-competition)
