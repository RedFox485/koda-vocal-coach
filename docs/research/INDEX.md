# Research Index — Koda Vocal Coach + Video Production Tools

Quick-find guide. All research from March 15-16, 2026 competition sprint.

---

## Competition & Demo Strategy
| Doc | What It Answers | Key Takeaway |
|-----|----------------|--------------|
| [DEMO-PLAN-REVIEW.md](DEMO-PLAN-REVIEW.md) | Will our demo win? What's weak? | Warm-up angle = 9/10. Deploy to Cloud Run. Don't claim 5 analyzers. |
| [COMPETITION-STRATEGY](../research/pipeline/COMPETITION-STRATEGY-RESEARCH.md) | How do serial winners approach competitions? | Blog post = +0.6 free points. Hook in 3s. Solo dev is a feature. |

## Product & UX
| Doc | What It Answers | Key Takeaway |
|-----|----------------|--------------|
| [VOCAL-WARMUP-RESEARCH.md](VOCAL-WARMUP-RESEARCH.md) | What do real vocal teachers do in warm-ups? | SOVT → scales → bridging. 5-10 min. Start easy, progress. |
| [GUIDED-SESSION-UX-RESEARCH.md](GUIDED-SESSION-UX-RESEARCH.md) | How should the warm-up UX feel? | 20-30% talk time. Fade-out coaching. First exercise = impossible to fail. |
| [VOCAL-ACCESS-GAP-RESEARCH.md](VOCAL-ACCESS-GAP-RESEARCH.md) | Why does this matter? Who needs it? | 46% of singers report disorders. Lessons nonexistent in most of world. |

## Video Production (Cadence & Voice)
| Doc | What It Answers | Key Takeaway |
|-----|----------------|--------------|
| [VIDEO-CADENCE-RESEARCH.md](cadence/VIDEO-CADENCE-RESEARCH.md) | How to pace a demo video? | 3-5s avg shot. 140 WPM narration. Pattern interrupt every 7-15s. |
| [VOICE-SERVICES-RESEARCH.md](pipeline/VOICE-SERVICES-RESEARCH.md) | Which TTS service for voiceover? | ElevenLabs (best quality) + Azure (best SSML control, huge free tier). |

## Reel (Browser Video Capture)
| Doc | What It Answers | Key Takeaway |
|-----|----------------|--------------|
| [BROWSER-VIDEO-CAPTURE-RESEARCH.md](pipeline/BROWSER-VIDEO-CAPTURE-RESEARCH.md) | How to capture web apps smoothly? | CDP BeginFrame + time virtualization. Never screen-record. |
| [SMOOTH-RECORDING-RESEARCH.md](pipeline/SMOOTH-RECORDING-RESEARCH.md) | Why do screen recordings stutter? | Real-time capture maxes at 10-20fps. Must decouple capture from video time. |
| [VIDEO-TOOLS-RESEARCH.md](pipeline/VIDEO-TOOLS-RESEARCH.md) | FFmpeg vs MoviePy vs Remotion? | FFmpeg subprocess for production. Manim for architecture animations. |

## CutRoom (Programmatic Video Editor)
| Doc | What It Answers | Key Takeaway |
|-----|----------------|--------------|
| [FFMPEG-DEEP-DIVE.md](../../cutroom-docs-link) | How to do X in FFmpeg? | 15 techniques with working commands. filter_complex_script for big graphs. |
| [NLE-ARCHITECTURE-RESEARCH.md](../../cutroom-docs-link) | How do Premiere/Resolve work internally? | Use OpenTimelineIO. ASS for overlays. FFmpeg over MLT. |
| [AUDIO-MIXING-RESEARCH.md](../../cutroom-docs-link) | How to mix audio professionally? | -14 LUFS. 5ms micro-crossfade at every edit. pedalboard for processing. |
| [PLATFORM-PUBLISHING-RESEARCH.md](../../cutroom-docs-link) | YouTube/TikTok upload APIs? | YT: 6 uploads/day. TikTok: PULL_FROM_URL. Soft subs for YT, burn-in for TikTok. |
| [VIDEO-TRANSFORMS-RESEARCH.md](../../cutroom-docs-link) | Aspect ratio, Ken Burns, blur-fill? | Working FFmpeg commands for all transforms. Avoid minterpolate. |
| [GPU-COST-ANALYSIS.md](../../cutroom-docs-link) | Cheapest way to render? | CPU: $0.001/video. vast.ai 3060: $0.003. Modal free tier: $0/mo for 1000 renders. |
| [CLOUD-RENDER-ARCHITECTURE.md](../../cutroom-docs-link) | Local preview + cloud render split? | Preview local (free). Final render on vast.ai. Batch mode saves provisioning cost. |

---

## CutRoom Docs (separate project)
All at `~/Documents/projects/cutroom/docs/`:
- `FFMPEG-DEEP-DIVE.md`
- `NLE-ARCHITECTURE-RESEARCH.md`
- `AUDIO-MIXING-RESEARCH.md`
- `PLATFORM-PUBLISHING-RESEARCH.md`
- `VIDEO-TRANSFORMS-RESEARCH.md`
- `GPU-COST-ANALYSIS.md`
- `CLOUD-RENDER-ARCHITECTURE.md`
- `CONTENT-AUTOMATION-VISION.md`
- `SHOT-SCORING-SYSTEM.md` (in `~/Documents/projects/reel/docs/`)
