# DevPost Submission — Koda Vocal Coach

**Copy-paste each section into the DevPost submission form.**

---

## Project Name
Koda — Real-Time Vocal Health Coach

## Tagline
AI-powered vocal coaching that detects strain and speaks technique cues at every breath point, powered by Gemini Live.

## About (Short Description)
Koda is a real-time vocal health coach that runs multiple acoustic analyzers in parallel to detect vocal strain, then uses a persistent Gemini Live session to speak technique cues at each breath point. Every singer in the world can now get real-time feedback that used to require a professional voice teacher.

## Inspiration
Hundreds of millions of people sing regularly — in choirs, bands, karaoke, practice rooms, showers. Almost none of them have ever worked with a vocal coach. Professional voice lessons cost $60-120/hour and require scheduling weeks in advance. Meanwhile, 46% of amateur singers report experiencing vocal disorders. The gap between "people who sing" and "people who get feedback" is enormous. We built Koda to close that gap.

## What it does
Koda listens to you sing through your phone's microphone and provides real-time vocal health coaching:

1. **Real-time strain detection**: Multiple acoustic analyzers (Parselmouth/shimmer/HNR, cepstral peak prominence, perceptual strain engine, phonation classifier) run in parallel on every audio frame with 96ms pipeline latency
2. **Adaptive baselines**: The strain engine learns YOUR voice — no fixed thresholds. Green zone means healthy phonation for you specifically
3. **Gemini Live coaching**: At every natural breath point, Gemini speaks a technique cue based on your current strain data, phrase duration, and vocal mode. Each cue is different — Gemini improvises within technique-anchored style guidelines
4. **Session persistence**: One continuous Gemini Live connection maintains full context. Every phrase, every zone transition, every coaching cue stays in memory. After each song, Gemini delivers a spoken summary of your entire session

## How we built it
- **Frontend**: Single-page web app with Canvas-based real-time visualizations (strain gauge, pitch tuner, vocal range mapper, session timeline)
- **Backend**: Python/FastAPI on Google Cloud Run. WebSocket streams raw audio at 10Hz from the browser
- **Signal Processing**: Parselmouth (Praat bindings) for shimmer and HNR. Custom CPP (cepstral peak prominence) detector for vocal fold closure. 8-channel perceptual strain engine. Phonation classifier with onset gating
- **Strain Engine**: Fuses multiple analyzer signals with weighted max/mean fusion. Adaptive CPP baseline tracks the singer's natural voice over time
- **Gemini Live**: Persistent streaming session via the Gemini Live API. Audio generation (native voice, not TTS). Strain data injected at phrase boundaries. System prompt enforces technique-anchored coaching style
- **Infrastructure**: Google Cloud Run (min-instances=1 for warm starts), Cloud Secret Manager for API keys

## Challenges we ran into
- **Baseline drift**: CPP (vocal fold closure quality) rises with volume. Asymmetric adaptation caused the baseline to ratchet toward loud frames, making quiet passages falsely appear strained. Fixed with symmetric adaptation at alpha=0.03
- **Canvas rendering on mobile**: DPR (device pixel ratio) scaling for Retina displays required syncing canvas backing stores to CSS container sizes at runtime, not hardcoded dimensions
- **Breath-point timing**: Gemini needs to speak between phrases, not during singing. Built a voiced-run counter with onset gating (300ms minimum) to detect natural pause points
- **96ms target**: Achieving sub-100ms pipeline latency required careful buffer management — 100ms audio frames, streaming WebSocket, no batch processing

## Accomplishments that we're proud of
- Real-time closed-loop coaching: singer strains → Koda detects it → Gemini speaks a technique cue → singer adjusts → gauge drops back to green. This feedback loop is the core product.
- 96ms pipeline latency from microphone to strain visualization
- Gemini improvises different coaching cues every time — reads the actual strain data, not pre-scripted responses
- Deployed and working on a real phone via Cloud Run — not a local demo
- Production-grade reliability: 50-request benchmark showed 119ms mean response time, 0 failures, 0 cold starts — even after 5 minutes idle, first request returns in 111ms

## What we learned
- Acoustic analysis of singing is fundamentally different from speech processing. Singing has intentional pitch variation, vibrato, and dynamic range that speech models treat as anomalies
- Adaptive baselines are essential — every voice is different, and even the same singer sounds different after warming up
- Gemini Live's persistent session context is powerful for coaching applications. The ability to reference "you were green for the first 8 phrases, then pushed too hard on the chorus" creates a coaching experience that feels genuinely aware

## What's next for Koda
- Song-specific coaching paths (warm-ups, scales, bridging exercises)
- Long-term progress tracking across sessions
- Integration with music education platforms
- Support for specific vocal techniques (vibrato development, breath support, register transitions)

## Built With
- Python
- FastAPI
- Google Cloud Run
- Gemini Live API
- Parselmouth (Praat)
- WebSocket
- Canvas API
- FFmpeg
- JavaScript

## Try it out
- **Live app**: https://koda-vocal-coach-358494904628.us-central1.run.app
- **Source code**: https://github.com/RedFox485/koda-vocal-coach
- **Blog post**: https://dev.to/redfox485/building-a-real-time-vocal-coach-with-gemini-live-and-signal-processing-3hbc
- **Demo video**: https://youtube.com/watch?v=iGvEHY5oIHA
