# YouTube Upload Metadata

## Title
Koda — Real-Time Vocal Coach Powered by Gemini Live

## Description
Koda listens to you sing, detects vocal strain in real-time using multiple parallel acoustic analyzers, and speaks technique cues at every breath point through a persistent Gemini Live session.

Built for the Gemini Live Agent Challenge.

🔗 Try it: https://koda-vocal-coach-358494904628.us-central1.run.app
📝 Technical deep-dive: [dev.to link — update after publishing]
💻 Source code: https://github.com/RedFox485/koda-vocal-coach

How it works:
• Browser captures mic audio over WebSocket at 10Hz
• Cloud Run backend runs Parselmouth (shimmer, HNR), CPP detector, perceptual strain engine, and phonation classifier in parallel
• 96ms total pipeline latency — strain gauge responds in real-time
• Adaptive baselines learn YOUR voice, not fixed thresholds
• Gemini Live speaks coaching cues at natural breath points — improvised, not scripted
• One persistent session — every phrase stays in context

Built with: Python, FastAPI, Google Cloud Run, Gemini Live API, Parselmouth/Praat, WebSocket, Canvas API

#GeminiLiveAgentChallenge #GeminiAPI #GoogleCloud #VocalCoach #AI

## Tags
Gemini Live, Gemini API, Google Cloud Run, vocal coach, AI, real-time audio, signal processing, Parselmouth, hackathon, demo, Gemini Live Agent Challenge

## Visibility
Public (or Unlisted — check competition rules)

## Category
Science & Technology
