# Morning Checklist — Koda Demo Video

**Last updated:** March 16, 2026 ~2:40 AM MDT by Koda (overnight build session)
**Deadline:** 6:00 PM MDT today

---

## What's Ready (Built Overnight)

- [x] **Koda backend** running with warm-up system prompt
- [x] **`/inject/ws` endpoint** — audio injection without session reset
- [x] **Playwright capture** tested — Watch mode + injection = UI responds
- [x] **9 voiceover clips** generated (Daniel voice, British broadcaster)
  - V1 in `audio/voiceover/` (original script)
  - V2 in `audio/voiceover_v2/` (warm-up angle, trimmed per review)
- [x] **Title card** rendered (`video/title_card.png`)
- [x] **Architecture diagram** exists (`docs/architecture.png`)
- [x] **Demo plan review** completed with actionable feedback (`docs/research/DEMO-PLAN-REVIEW.md`)
- [x] **Vocal warm-up research** completed (`docs/research/VOCAL-WARMUP-RESEARCH.md`)
- [x] **Gemini system prompt** updated for warm-up coaching angle
- [x] **Gemini greeting** updated: "Let's warm up your voice"
- [x] **Green zone coaching** added — Gemini gives positive reinforcement every 4th green phrase
- [x] **CutRoom** — assembler, mixer, overlay, exporter, preview, cloud render, CLI all built
- [x] **Reel** — inject, capture, voiceover generation all built

## What Daniel Needs to Do

### 1. Deploy to Cloud Run (CRITICAL — disqualification risk)
The demo review flagged this as #1 priority. Fly.io ≠ Google Cloud.

**Status:** Dockerfile exists and is Cloud Run-ready. GCP project: `gen-lang-client-0999911778`. gcloud CLI is installed and authenticated as `dannyfordsystems@gmail.com`.

**Step 1: Enable Cloud Run API** (must do in browser — can't automate):
Open: https://console.developers.google.com/apis/api/run.googleapis.com/overview?project=gen-lang-client-0999911778
Click "Enable". Wait 1-2 minutes.

**Step 2: Enable Artifact Registry** (for Docker images):
```bash
gcloud services enable artifactregistry.googleapis.com --project=gen-lang-client-0999911778
```

**Step 3: Deploy**:
```bash
cd ~/Documents/projects/koda-comp-gemini

# Set the Gemini API key as a secret
echo -n "$(security find-generic-password -s GEMINI_API_KEY -w)" | \
  gcloud secrets create gemini-api-key --data-file=- --project=gen-lang-client-0999911778 2>/dev/null || \
  echo -n "$(security find-generic-password -s GEMINI_API_KEY -w)" | \
  gcloud secrets versions add gemini-api-key --data-file=- --project=gen-lang-client-0999911778

# Build and deploy (takes ~5-10 minutes first time)
gcloud run deploy koda-vocal-coach \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" \
  --session-affinity \
  --min-instances 0 \
  --max-instances 1 \
  --timeout 300 \
  --port 8080 \
  --project gen-lang-client-0999911778
```

**Step 4: Test**: Visit the Cloud Run URL that's printed after deploy. Verify the UI loads and Gemini greets.

### 2. Record DJI Hook Shot (~5 minutes)
- Close-up: face + phone, Koda UI visible on screen
- Warm lighting, mouth open and singing
- 30+ seconds of footage, we'll pick the best 3 seconds
- **Tip from review:** Close shot > wide shot. Human emotion > technical proof.

### 3. Choose Singing Audio
**Option A: Daniel sings** — most authentic, warm-up exercises + a song
**Option B: Professional vocal stem** — agent searching overnight (check `audio/singing/`)
**Option C: Use calibration_AB.wav** — already tested, proven strain contrast

For the warm-up angle, Option A is best: Daniel does actual warm-up exercises (humming, scales, then a song). This matches the narrative perfectly.

### 4. Test Recording
```bash
# Start backend (if not running)
cd ~/Documents/projects/koda-comp-gemini
GEMINI_API_KEY=$(security find-generic-password -s "GEMINI_API_KEY" -w) \
  .venv/bin/python -m uvicorn src.vocal_health_backend:app --port 8000

# Open browser: http://localhost:8000
# Click "Enable Microphone"
# Sing — verify gauge responds, Gemini coaches
```

### 5. Screen Record (~30 minutes for 3+ takes)
Record the full warm-up flow:
1. Click Start → Gemini greets with warm-up intro
2. Easy humming/singing → gauge GREEN
3. Push harder → gauge shifts YELLOW
4. Gemini coaching cue → pause → adjust → back to GREEN
5. "You're warm" → sing a song verse + chorus
6. Stop → 4 seconds silence → Gemini summary

**Recording tool:** macOS Cmd+Shift+5 or OBS Studio
**Record at:** 1920x1080 native, include system audio (Gemini's voice)

### 6. Assemble with Pipeline
```bash
# Quick version (manual in iMovie if pipeline isn't ready):
# 1. Import DJI clip + screen recording into iMovie
# 2. Trim and arrange per shot list
# 3. Add voiceover clips from audio/voiceover_v2/
# 4. Add title card
# 5. Export

# Pipeline version (if CutRoom is working):
cd ~/Documents/projects/cutroom
python cutroom.py render --timeline ../koda-comp-gemini/config/demo_timeline.json
```

### 7. Upload + Submit
- Upload to YouTube (public or unlisted)
- Submit on DevPost by 5:00 PM MDT
- Include: video URL, GitHub repo, blog post link (bonus points)

---

## Critical Items from Demo Review

1. **Don't claim "5 parallel analyzers"** — wavelet scattering isn't running. Say "multiple acoustic analyzers" or "our analysis pipeline"
2. **Deploy to Cloud Run** — non-negotiable for submission
3. **Let Gemini be heard** — don't talk over coaching cues. 50/50 voiceover vs product audio.
4. **The correction moment is key** — singer adjusts → gauge drops to green. Give it room.
5. **Warm-up = reframe, don't rebuild** — system prompt already updated. No new code needed.

---

## Files to Know About

| What | Where |
|------|-------|
| Voiceover clips (warm-up) | `audio/voiceover_v2/` |
| Voiceover clips (original) | `audio/voiceover/` |
| Demo plan | `docs/DEMO-VIDEO-PLAN.md` |
| Demo review (read this!) | `docs/research/DEMO-PLAN-REVIEW.md` |
| Warm-up research | `docs/research/VOCAL-WARMUP-RESEARCH.md` |
| Title card | `video/title_card.png` |
| Architecture diagram | `docs/architecture.png` |
| Capture screenshots | `video/captures/pipeline2_*.png` |
| Updated demo script | `config/demo_script_warmup.json` |
| Audio pauses config | `config/audio_pauses.json` |
