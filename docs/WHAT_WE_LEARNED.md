# Competition Reflection: Gemini Live Agent Challenge

> **Read this before every future competition.** This is our distilled competition knowledge from the Koda Vocal Coach submission to the Gemini Live Agent Challenge, March 2026.

---

## The Result

- **Submitted**: March 16, 2026, 5:51 PM MDT (9 minutes before deadline)
- **Category**: The Live Agent
- **All bonus points claimed**: Published content (+0.6), automated deployment (+0.2), GDG profile (+0.2)
- **DevPost**: https://devpost.com/software/koda-real-time-vocal-health-coach

---

## Time Breakdown

| Phase | Time Spent | Notes |
|-------|-----------|-------|
| **Core product** (backend + frontend) | ~40 hrs (prior week) | FastAPI, Gemini Live, signal processing, Canvas UI |
| **Demo video pipeline** (Reel + CutRoom) | ~12 hrs overnight | Built both tools from scratch |
| **Gemini audio harvesting** | ~1 hr | 2 runs, 16 clips, 4 selected |
| **Multi-take Reel capture** | ~1 hr | 2 takes, scored, best selected |
| **CutRoom assembly + debugging** | ~3 hrs | Mixer bugs, overlay failures, timing fixes |
| **DJI hook shot** | ~1.5 hrs | USB incompatibility debugging, wireless workaround |
| **Written deliverables** | ~2 hrs | Blog (2,036 words), DevPost text, YouTube metadata |
| **Singer search** | ~45 min | **WASTED** — should have decided earlier |
| **Video iterations** | ~1.5 hrs | v1 → v2 → v3 → v4, each with fixes |
| **Submission** (YouTube, dev.to, DevPost) | ~45 min | YouTube API setup was worth it |
| **Infrastructure** (GitHub, deployment proof) | ~30 min | Should have done this earlier |

**Total competition day**: ~10 hours of focused execution

---

## Where We Lost Time (Time Creep)

### 1. Singer Selection — 45 min wasted
We spent 45 minutes trying to find a different singer (male, non-autotuned, CC0). Every source either blocked automated downloads, required login, or returned unsuitable tracks.

**Lesson**: Make creative asset decisions EARLY and commit. If the asset works technically (triggers strain gauge correctly), ship it. Judges don't care about the singer's vocal timbre — they care about the tech demo.

**Rule for next time**: Lock creative assets 4+ hours before deadline. No changes after that.

### 2. DJI USB Debugging — 1.5 hrs
The DJI Pocket 2 has a known firmware bug with M1 Macs. We tried: both USB ports, Apple cable, sleep/wake, driver research, DJI Mimo desktop app, SD card extraction. Final fix: wireless transfer via DJI Mimo phone app.

**Lesson**: When hardware doesn't work in 10 minutes, skip to the workaround. Don't debug firmware bugs during a competition.

**Rule for next time**: Budget 0 minutes for hardware debugging. If it doesn't work immediately, use the phone.

### 3. CutRoom Overlay Failures — 30 min
FFmpeg lacked libass. We tried 3 different approaches before accepting overlays were impossible.

**Lesson**: Know your tool limitations BEFORE the competition day. Run a smoke test of every CutRoom feature the night before.

**Rule for next time**: Pre-flight checklist for all tools 24 hours before deadline.

### 4. Late Discovery of Required Deliverables — Cost us stress
We didn't discover the "Proof of Cloud Deployment" (separate recording) requirement until 2.5 hours before deadline. The GitHub repo requirement was also caught late.

**Lesson**: Read the FULL competition rules on day 1. Create a checklist of every required deliverable immediately.

**Rule for next time**: First hour of any competition = read rules, create deliverable checklist, work backward from deadline.

### 5. Video Duration Mismatch Bug — 30 min
CutRoom's assembler uses source durations, not timeline durations. Shot8a had 2s of source mapped to 8s of timeline, silently truncating the last 18 seconds including the closing voiceover.

**Lesson**: Always verify the rendered video duration matches expectations. Watch the last 30 seconds.

**Rule for next time**: QA checklist includes "watch first 10s and last 10s of every render."

---

## What Went Right (Highlight Moments)

### 1. Building Reel + CutRoom Overnight
We built two production tools (programmatic browser capture + video editor) in a single overnight session. These tools are now reusable for every future project. The ROI on this investment is enormous.

### 2. Gemini Audio Harvesting Strategy
Instead of trying to capture Gemini audio live during screen recording (unreliable), we harvested clips separately and placed them in the CutRoom timeline. This gave us perfect content, timing, and multiple options to choose from. Brilliant separation of concerns.

### 3. YouTube API Upload
Setting up OAuth and building the upload script took 15 minutes and saved us from manual upload. More importantly, it's now a **permanent capability** — every future DannyMakesThings video is a one-command upload.

### 4. dev.to API Publishing
Same story. One API call published a 2,000-word technical blog. Permanent capability for the publishing flywheel.

### 5. Speed Test as Marketing Asset
Running 50 requests and documenting "119ms avg, 0 failures, 0 cold starts" gave us a concrete, impressive performance claim. We added it to the blog, DevPost, and it may catch judges' eyes.

### 6. The "Ship Now, Improve Later" Mindset
Multiple times we almost went down rabbit holes (perfect singer, tap animation, background music). Each time, the discipline of "submit first, improve if time" kept us on track. We submitted with 9 minutes to spare instead of missing the deadline chasing perfection.

---

## Aha Moments

1. **"The closing voiceover was silently cut off"** — CutRoom's source/timeline duration mismatch was invisible until we watched the video. Automated QA (duration check, last-frame extraction) would have caught this instantly.

2. **"The voiceover should start immediately"** — Daniel's instinct that silence at the start feels broken was exactly right. Judges will evaluate within the first 3 seconds. Audio from frame 1.

3. **"Read the rules on day 1"** — We almost missed the Cloud deployment proof, the GitHub repo requirement, and the blog hashtag. Each of these is a pass/fail gate.

4. **"Competition bonus points are free money"** — Blog (+0.6), deploy script (+0.2), GDG profile (+0.2) = 1.0 bonus points for ~30 minutes of work. That's the highest ROI time spent in the entire competition.

5. **"Every tool you build is equity"** — Reel, CutRoom, YouTube uploader, dev.to publisher — none of these were "competition work." They're permanent infrastructure that compounds across every future project.

---

## The Competition Playbook (Use This Next Time)

### Week Before
- [ ] Read FULL competition rules — create deliverable checklist
- [ ] Identify all required fields (DevPost, etc.) and draft them
- [ ] Smoke test all tools (CutRoom render, Reel capture, FFmpeg features)
- [ ] Set up GitHub repo, README with spin-up instructions, deploy script
- [ ] Lock creative assets (singer, music, visual style)

### Competition Day — First Hour
- [ ] Verify live deployment works (speed test)
- [ ] Record Cloud deployment proof
- [ ] Verify all bonus point opportunities

### Competition Day — Middle
- [ ] Build demo video using Reel + CutRoom pipeline
- [ ] Write blog post and DevPost text
- [ ] QA: watch full video, check duration, verify closing line plays

### Competition Day — Last 2 Hours
- [ ] Upload video to YouTube (use `scripts/youtube_upload.py`)
- [ ] Publish blog to dev.to (use API)
- [ ] Submit DevPost with all links and attachments
- [ ] Claim ALL bonus points (blog, deploy script, GDG)
- [ ] Final 30 min = buffer for fixes + re-upload if needed

### Never Do During Competition
- [ ] Debug hardware (use workaround immediately)
- [ ] Search for creative assets (lock these beforehand)
- [ ] Rebuild FFmpeg or install system dependencies
- [ ] Change the singer/music within 2 hours of deadline

---

## Tools Built (Reusable)

| Tool | Location | What It Does |
|------|----------|-------------|
| **Reel** | `~/Documents/projects/reel/` | Programmatic browser capture with Playwright, multi-take scoring |
| **CutRoom** | `~/Documents/projects/cutroom/` | JSON timeline → FFmpeg render, audio ducking, multi-track mixing |
| **YouTube Uploader** | `scripts/youtube_upload.py` | One-command YouTube upload via API (OAuth cached) |
| **dev.to Publisher** | API key in Keychain | `curl` or Python to publish blog posts instantly |
| **Gemini Harvester** | `scripts/harvest_gemini_audio.py` | Capture Gemini Live audio responses as WAV files |
| **Capture Script** | `scripts/capture_demo.py` | Full Reel capture + audio injection pipeline |
| **Speed Tester** | (inline bash) | 50-request benchmark with cold-start simulation |

---

## Upcoming Competitions Worth Entering

### Microsoft AI Agents Hackathon (April 2026)
- **Deadline**: April 30, 2026
- **Prize**: $10K per category winner + Microsoft Build 2026 ticket
- **Fit**: HIGH — we just built an AI agent. Koda could be adapted or we build something new with Azure.
- **URL**: https://developer.microsoft.com/en-us/reactor/events/26647/

### Automation Innovation Hackathon 2026
- **Prize**: $15K first place ($22.5K total)
- **Fit**: MEDIUM — could build an automation tool using our pipeline
- **URL**: https://automation-innovation.devpost.com/

### DevNetwork AI + ML Hackathon (May 11, 2026)
- **Prize**: Conference passes + Amazon devices
- **Fit**: MEDIUM — good for portfolio/visibility even if prizes are smaller

### Anthropic Claude Code Hackathons (watch for next one)
- **Last round**: Feb 2026, $100K in API credits, 4% acceptance rate
- **Fit**: EXTREMELY HIGH — Daniel literally builds everything with Claude
- **Watch**: https://devpost.com/hackathons (filter: Anthropic)

### What to Watch For
- Google often runs seasonal hackathons (I/O, Cloud Next, DevFest)
- AWS re:Invent hackathons typically announced ~September
- Keep DevPost notifications on for AI/Cloud categories

---

## Final Thought

> We built a real-time vocal coach, two video production tools, a YouTube upload pipeline, and a blog publishing system — then submitted everything with 9 minutes to spare. The product is real, the demo is real, the deployment is real. Whatever the judges decide, we built something worth building.

*— Koda + Daniel, March 16, 2026*
