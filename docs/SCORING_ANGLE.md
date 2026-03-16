# Scoring Angle — Gemini Live Agent Challenge

**Last updated**: March 15, 2026
**Deadline**: March 16, 2026 5:00 PM PDT (6:00 PM MDT)

---

## Competition Intelligence

### Judges
- **DevRel people, NOT researchers** — they care about user experience, not code elegance
- Kelvin Boateng (Google judge): "We're not looking at the best-written code — we're looking at the end-user experience"
- Logan Kilpatrick — leads Google DeepMind DevRel for Gemini, AI Studio, Veo, Imagen

### Google's Marketing Angle
**"Gemini doesn't just hear words — it hears YOU."**

Google is promoting **native audio perception** as the killer differentiator over competitors. They want demos that show the model HEARING nuance (tone, emotion, pace), not just transcribing words.

Official tagline: "Stop typing, start interacting!"

### What They Want for NEXT 2026 Marketing
Winners present at Google Cloud NEXT (April 22-24, Las Vegas). The conference theme is **agentic AI** — production-ready agents solving real problems. Google wants to show the NEXT audience that their developer community is building real, deployable agents — not hackathon toys.

---

## Scoring Breakdown

| Criterion | Weight | What Wins | Our Strategy |
|-----------|--------|-----------|-------------|
| **Innovation & "Beyond Text" UX** | **40%** | Break the chatbot paradigm. Show Gemini HEARING nuance | Gemini coaches based on **vocal quality**, not transcription. It hears strain, breathiness, fatigue |
| **Technical Implementation** | 30% | Clean architecture, GCP integration, error handling, hallucination avoidance | Cloud Run deploy + architecture diagram + clean WS contract + EARS as proprietary tool |
| **Demo & Presentation** | 30% | Hook in first 15 seconds, polished UX, clear problem→solution, architecture diagram | "50M self-taught singers risk injury with zero feedback" → live demo of real-time detection |

### Bonus Points (up to +1.0 on 5-point scale = 20% boost)
| Bonus | Points | Status |
|-------|--------|--------|
| Blog post (#GeminiLiveAgentChallenge) | +0.6 | TODO — quick dev.to post tonight |
| Automated deployment scripts in repo | +0.2 | DONE — deploy-gcp.sh |
| Google Developer Group membership | +0.2 | TODO — 2 min signup |

---

## Why We Win

### Our Competitive Edge
Most competitors will build **chatbots with voice input** — basically text-box apps with a mic button. We're building something that uses the audio modality for its **core function**:

- Gemini perceives vocal health through **sound quality**, not words
- Real-time strain detection from acoustic properties (tension, breathiness, harmonics)
- Coaching at phrase boundaries — Gemini speaks only when it has something useful to say
- The "Beyond Text" factor is our entire product, not a feature

### Past Winner Patterns (2024 Gemini Competition)
1. **Accessibility/health angle** — 4 of 8 winners had impact stories
2. **Real-time interaction** — not batch processing
3. **Made Gemini look versatile** — showcased different capabilities
4. **Polished UX over deep tech** — judges rewarded finished products

### Our Pitch
**One-liner**: "Every tool that measures vocal health costs thousands and requires a clinician. Every app a singer can afford is just a pitch tuner. We're the first real-time vocal health monitor anyone can use while they're actually practicing."

**The kids angle** (impact story): Self-taught singers learning on YouTube have ZERO health feedback. Kids who can't afford lessons risk injury with no warning. No existing consumer tool fills this gap.

**Market gap**: We sit between Smule ($0, pitch only) and MDVP ($6,000, clinical). That gap is empty.

---

## Demo Video Strategy (4 min max)

### Structure (judges watch video FIRST, before reading anything)

| Time | Content | Purpose |
|------|---------|---------|
| 0:00-0:15 | Problem hook: "50 million self-taught singers risk vocal injury every day with zero feedback" | Hook — judges decide in 15 seconds |
| 0:15-0:30 | Market gap visual: pitch tuners ($0) ←→ clinical tools ($6K). "This gap is empty." | Establish the opportunity |
| 0:30-2:30 | **LIVE DEMO**: Daniel sings, strain meter moves, Gemini coaches in real-time | The money shot — 2 full minutes of working product |
| 2:30-3:00 | Architecture diagram + Gemini integration explanation | Technical credibility (30% of score) |
| 3:00-3:30 | Show key moments: yellow warning → Gemini speaks → singer adjusts → green | The "aha" moment |
| 3:30-3:50 | Impact: "accessible to anyone with a phone and a voice" | Emotional close |
| 3:50-4:00 | Logo + links | CTA |

### Key Moments to Capture
1. **The strain spike**: Singing a high note, meter goes yellow/red, Gemini intervenes
2. **The recovery**: Singer eases off, meter returns to green, Gemini praises
3. **The vibrato detection**: Badge lighting up during a sustained note with vibrato
4. **The coaching audio**: Gemini's voice coming through with specific feedback

---

## Submission Checklist

| Required | Status | Notes |
|----------|--------|-------|
| Public code repo + README | TODO | Setup/deployment instructions |
| Google Cloud deployment proof | TODO | Cloud Run deploy tonight |
| Architecture diagram | TODO | Visual system diagram |
| Demo video (max 4 min) | TODO | Record tonight |
| Text description | TODO | Features, tech, data, learnings |
| Category: Live Agents | Ready | Perfect fit |

---

## Technical Proof Points for Judges

1. **Native audio perception** — EARS system processes raw audio, not transcription
2. **Real-time streaming** — WebSocket bidirectional, 100ms analysis frames
3. **Gemini Live API** — persistent bidi session, voice coaching at phrase boundaries
4. **Session-adaptive baselines** — strain detection calibrates to each singer's voice
5. **Production architecture** — FastAPI + WebSocket + Cloud Run, min-instances=1
6. **Feedback ladder** — green=silent, yellow=text, red=voice coaching (not noisy)
