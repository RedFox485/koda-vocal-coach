# Demo Plan Review — Brutally Honest Competition Judge Assessment

**Reviewer posture:** Google DevRel judge, 47th submission in the queue, 3 hours in.
**Date:** March 16, 2026

---

## 1. Hook Strength (First 10 Seconds)

**Score: 8/10 — Strong, with one risk.**

The DJI Osmo shot is genuinely differentiating. Most entries will open with a screen recording or a slide. A real human singing into a phone with a moving strain gauge is viscerally different. It earns the "pattern interrupt" label.

The risk: 3 seconds of DJI footage with no voiceover and no text overlay could read as confusing rather than intriguing. A judge skimming at 1.5x speed might not register what the phone screen is showing. The shot needs to be tight enough that the singing is obvious — mouth open, energy visible — within the first frame.

**The warm-up angle change helps here.** "Ready for your warm-up?" as the Gemini greeting immediately after the hook establishes a teacher-student dynamic. This is stronger than "Ready when YOU are!" because it tells the judge what this IS, not just that it exists.

**Actionable:** Make sure the DJI shot is CLOSE — face and phone, not a wide shot of a room. The phone screen being visible matters less than the human emotion of singing being immediate.

---

## 2. Product Clarity

**Score: 7/10 — Clear product, slightly muddled positioning.**

By 0:30, a judge will understand: "This is an app that listens to you sing and coaches you in real time using Gemini." That is clear.

What is NOT immediately clear with the warm-up angle: is this a vocal health tool (prevent injury) or a singing coach (improve technique)? The existing plan sells "vocal health" — preventing strain, detecting injury risk. The new warm-up angle sells "vocal coaching" — guided exercises, technique improvement. These are different value propositions, and mixing them weakens both.

**The warm-up angle actually makes this stronger for the competition**, because "guided warm-up with adaptive coaching" is a more complete product story than "passive strain meter." But you need to pick one framing and commit. I would frame it as: "Koda is your AI vocal coach. It guides warm-ups, detects strain in real time, and coaches you through every phrase — powered by Gemini Live." That unifies both angles.

**Actionable:** The problem statement should be about the lack of accessible coaching (not just the lack of strain detection). "Hundreds of millions of singers practice alone with zero feedback" is better than "nobody tells them when they're hurting their voice." The first framing includes the warm-up angle; the second is only about injury prevention.

---

## 3. The Warm-Up Angle

**Score: 9/10 — This is the right call. This is what makes it feel like a product.**

Here is why the warm-up angle is significantly better than passive monitoring:

1. **It gives Gemini agency.** In the old plan, Gemini reacts. In the warm-up plan, Gemini LEADS. This makes Gemini look more capable. Google judges want to see Gemini doing impressive things. An AI that guides a structured lesson is more impressive than an AI that comments on what just happened.

2. **The closed feedback loop is real.** Gemini says "try this" > singer tries > gauge shows result > Gemini responds to the result. This is a genuine multi-turn, multimodal interaction loop. Most entries will be "user says thing > AI responds." This is "AI leads > human performs > AI perceives result > AI adapts." That loop is the entire pitch for Gemini Live.

3. **It structures the demo.** The warm-up gives you a natural narrative arc: easy exercise (green) > harder exercise (yellow) > coaching correction > singer adjusts > green again > "you're warm, sing something!" > real song with coaching. This arc is built into the product, not imposed by editing.

4. **It answers the "is this real?" question.** A passive strain meter could be faked with a color-changing gauge on a timer. A warm-up where Gemini guides specific exercises and responds to what it hears is much harder to fake. It proves the system works.

**One concern:** You need Gemini to actually guide exercises. Is this built? The session state from March 15 shows Gemini greeting and phrase-boundary coaching, but not structured warm-up guidance. If this is a new feature that needs to be implemented today, that is a significant risk given the deadline is today.

**Actionable:** If the warm-up flow is not already implemented, do NOT build it from scratch. Instead, reframe what you already have: the greeting says "Let's warm up" and the singer starts with easy singing (which you already show as green zone). Then pushes harder (yellow). Then Gemini coaches. The "warm-up" is just the existing flow with a different framing in the greeting and voiceover. Do not build new features on deadline day.

---

## 4. Technical Credibility

**Score: 7/10 — The depth is real, but the delivery needs calibration.**

Naming Parselmouth, CPP, wavelet scattering, and the 8-channel perceptual engine by name is the right call. This is a Google DevRel audience — they know what cepstral peak prominence is, or at least they know it is not something you Google in 5 minutes. It signals depth.

**The risk is information overload.** The green zone shot has 52 words of voiceover naming 5 analyzers with technical descriptions. That is a LOT for 22 seconds. A judge who does not know audio signal processing will zone out. A judge who does will be impressed but may not retain it.

**The 96ms latency is your best technical proof point.** Lead with it, do not bury it in a list. "Total pipeline latency: 96 milliseconds" is the one number a non-specialist judge can understand and be impressed by. The algorithm names are for the specialist judges who will pause on the architecture diagram anyway.

**Actionable:** Trim the green zone voiceover. Instead of naming every analyzer, say: "Five acoustic analyzers run in parallel — measuring vocal fold closure, harmonic stability, and timbral strain. Total pipeline latency: 96 milliseconds. The baseline adapts to YOUR voice." That is 28 words instead of 52. Save the full algorithm names for the architecture diagram shot where they appear visually and the pace is deliberately faster.

**On wavelet scattering:** Your session state says kymatio is NOT available and scatter_score = 0 always. Do not claim wavelet scattering in the demo if it is not running. Judges who check the code will notice. Mention it in the architecture diagram as a component, but do not claim it is active in the pipeline if it is not. This is the kind of thing that turns a win into a disqualification or credibility hit.

---

## 5. The "Wow" Moment

**Score: 8/10 — The correction moment is strong. But the BIGGER wow is the warm-up progression.**

The gauge going from yellow back to green after the singer adjusts is a good moment. It is visual proof of a closed feedback loop.

But with the warm-up angle, the bigger wow is the SEQUENCE: Gemini guides an easy exercise (green) > guides a harder exercise (singer pushes, yellow) > "I heard tension, drop your jaw" > singer adjusts > green > Gemini says "Better. You're warm." That is not just one moment — it is a 60-second story of an AI teaching a human, perceiving the result, and adapting. That story is worth more than any single gauge-color change.

**The wow moment most entries will NOT have:** Gemini's voice coaching another human voice in real time. Every other entry will be text-to-speech or Gemini talking to a user about text/images. Yours has Gemini literally coaching a singer MID-PERFORMANCE. That is the Beyond Text Factor. Make sure you give it room to breathe — do not talk over it.

**A potentially bigger wow you are missing:** The session summary at the end. If Gemini says something like "Strong warm-up. Your tension came in on the third exercise around the A4 — you corrected it nicely. Your song was mostly clean with a little push on the chorus belt. Good session." — that proves full session context retention. Make sure the summary references specifics that happened during the demo. If it is generic, it hurts more than it helps.

**Actionable:** Script the warm-up progression as the MAIN wow, not just the single correction moment. The arc is: easy (green) > challenging (yellow) > correction (green) > "you're warm" > song > summary. Every step proves a different capability.

---

## 6. Emotional Arc

**Score: 7/10 — Good bones. The warm-up angle makes it better. Two weak spots.**

The current arc: Curiosity (DJI hook) > Understanding (how it works) > Awe (Gemini coaching) > Purpose (why it matters).

With the warm-up angle, it becomes: Curiosity (DJI hook) > Engagement (Gemini greets, warm-up begins) > Tension (strain builds) > Resolution (singer corrects) > Joy (singing with a coach) > Reflection (session summary) > Purpose (why it matters).

That is a more complete emotional arc. The resolution moment (correction > green) provides genuine catharsis.

**Weak spot 1:** The architecture diagram (Shot 7) kills the emotional momentum. You go from Gemini's warm session summary straight into rapid-fire technical jargon. That is a tonal whiplash. Consider moving the architecture shot BEFORE the song summary, so the emotional arc flows: technical proof > song > summary > impact close. Or keep it where it is but acknowledge the shift: "Here is how it works" as a clean transition.

**Weak spot 2:** The AI voiceover. A calm male AI voice reading a script is competent but not memorable. The most emotionally resonant demos from past winners (Gaze Link, ViddyScribe) featured real human narration — imperfect, but genuine. An AI voice signals "I generated this" which slightly undermines the "this is a real product for real people" message.

**Actionable on voiceover:** If you have time, record yourself narrating. Your genuine excitement about this project will come through in ways ElevenLabs cannot replicate. If you do not have time or are not comfortable, the AI voice is fine — it will not lose the competition. But it will not win the "authentic" points that a real voice would.

---

## 7. Pacing

**Score: 8/10 — 2:45 is right. Could be tighter at 2:30.**

2:45 is within the sweet spot. Judges reviewing dozens of entries will appreciate it being under 3 minutes. Past winners averaged 2-2.5 minutes.

The potential fat:
- Shot 4 (green zone) at 28 seconds is long for "everything is fine." If you trim the voiceover as suggested above (52 words to ~28), you save 10 seconds.
- Shot 5c (second coaching cue) at 20 seconds could be 12-15 seconds. The point is made quickly — "the second cue is different."
- Shot 6 (summary) at 27 seconds is appropriate only if Gemini's summary is genuinely impressive. If it is generic, cut it to 15 seconds.

**With the warm-up angle, the pacing changes:**
- Warm-up exercises need ~50 seconds (3-4 exercises with progression)
- Song singing with coaching needs ~40 seconds
- Summary needs ~15-20 seconds
- Architecture + close needs ~35 seconds

That puts you at ~2:30, which is ideal.

**Actionable:** Aim for 2:30. Every second saved is a judge's patience preserved.

---

## 8. What is MISSING?

**Three things a Google DevRel judge will look for:**

1. **Proof this runs on Google Cloud.** The rules require "backend must be hosted on Google Cloud" and "must demo proof of this." Your session state shows GCP Cloud Run is "prepped but not yet deployed." Fly.io is not Google Cloud. You MUST deploy to Cloud Run before submitting. A 2-second screen flash of the Cloud Run console showing the service running would satisfy this. Without it, you could be disqualified.

2. **Barge-in / interruption handling.** The Live Agent category specifically calls out "handles interruptions (barge-in) gracefully." The warm-up angle actually showcases this naturally — the singer interrupts Gemini by starting to sing, and Gemini stops talking and starts listening. Make sure this moment is visible in the demo. If the singer starts singing while Gemini is still coaching, Gemini should gracefully stop. This is a category requirement.

3. **A before/after comparison.** Past winning demos showed the "before" state — the painful way things are done now — before showing the solution. You have the problem statement in words ("nobody tells them when they're hurting their voice") but not visually. A 3-second clip of a singer massaging their throat after practice, or a stock image of someone with a hoarse voice, would make the problem visceral. This is optional but effective.

**Also missing but lower priority:**
- User testing data ("we tested with 5 singers and..." — judges love this)
- Multi-language angle (you mention "in any language" in the close but never demonstrate it)
- Mobile responsiveness (if the UI is only desktop, that limits the "any singer anywhere" pitch)

---

## 9. What Would Make Me NOT Pick This as a Winner?

**Being brutally honest — here are the dealbreakers:**

1. **If Gemini's coaching sounds generic or scripted.** If every coaching cue is "ease up on the push" and the session summary is "good session, you did well" — the magic is gone. The demo lives or dies on Gemini saying something that could ONLY have been said based on what it just heard. If a judge suspects the responses are pre-recorded, you lose.

2. **If the strain gauge does not visibly respond to the singing.** If the gauge lags, stays static, or flickers randomly, judges will assume it is decorative. The gauge must move in sync with audible changes in the singing. The green-to-yellow transition must be OBVIOUS — not subtle.

3. **If the voiceover carries the demo instead of the product.** The best demos let the product speak. If 70% of the audio is voiceover and 30% is the product working, the balance is wrong. It should be closer to 50/50. Let Gemini talk. Let the singing be heard. Let the gauge be seen. The voiceover should fill gaps, not dominate.

4. **If it looks like a hackathon project, not a product.** The UI redesign work from March 15 sounds polished. But if there are visible bugs, loading spinners that hang, console errors, or layout shifts during the recording — that is what judges will remember.

5. **If the claim of "5 parallel analyzers" does not match reality.** Kymatio/scatter is not running. Phonation classifier depends on scatter features and is also not running. That means you have 3 working analyzers (Parselmouth shimmer/HNR, CPP, EARS v11), not 5. Saying "5 parallel analyzers" when only 3 are active is a credibility risk if a judge reads the code. Say "multiple acoustic analyzers" or "our analysis pipeline" instead of committing to a number that does not match.

6. **Not deployed on Google Cloud.** This is not a quality issue — it is a disqualification risk. Deploy to Cloud Run today.

---

## 10. Comparison to Past Winners

**Versus Jayu (Best Overall):**
Jayu showed Gemini controlling applications on screen — multimodal in the "vision" sense. Koda shows Gemini perceiving and coaching via audio — multimodal in the "audio" sense. Both are genuine Beyond Text demonstrations. Koda's advantage: it is a more focused product. Jayu was broad (browser control, music, games). Koda solves one problem deeply. For the Live Agent category specifically, Koda is a stronger fit because it is a sustained, interactive voice session — not episodic commands.

**Versus ViddyScribe (Best Web):**
ViddyScribe had a clear, specific accessibility angle: blind people experiencing video. Koda has a clear, specific health/access angle: singers getting coaching. Both have the "helping underserved people" quality Google rewards. ViddyScribe's advantage: its problem is easier to understand in 5 seconds. "Blind person watches a video" is instant empathy. "Singer might hurt their voice" requires a beat more context. Koda's advantage: it is LIVE and INTERACTIVE, not batch processing. Real-time beats async for the Beyond Text Factor.

**Versus Gaze Link (Best Android):**
Gaze Link's advantage was undeniable emotional impact — ALS patient typing with their eyes. Koda cannot compete on that level of emotional weight. Gaze Link had quantifiable impact (85% fewer keystrokes). Koda does not have user testing data with quantified improvement. This is a gap.

**Overall competitive position:** Koda is in the top tier of what this competition is looking for. It is a genuine multimodal agent, not a chatbot wrapper. The real-time audio perception is exactly what Google wants to showcase. It sits comfortably alongside the past winner profiles.

**What separates Koda from the pack:** Most entries in the Live Agent category will be voice-controlled assistants — "Hey Gemini, do this thing." Those are glorified voice commands. Koda is Gemini perceiving continuous audio, extracting meaning from sound quality (not words), and coaching in real time. That is fundamentally different and significantly harder to build. If judges recognize that distinction, Koda wins on innovation.

---

## 11. The Access Angle

**Score: 8/10 — It lands. One tweak.**

"A voice lesson costs $80/hour — if you can find a teacher at all" is effective as a closer. It works because:
- It quantifies the barrier (judges understand money)
- The "if you can find a teacher at all" widens it from cost to access (rural areas, developing countries)
- It comes AFTER the demo, so it is earned (you showed the tech first)

**The tweak:** "For free" at the end might raise skepticism. Judges know this uses Gemini API calls which cost money. "For the cost of an internet connection" or just dropping "for free" and ending with "Koda gives every singer a real-time vocal health coach. Powered by Gemini Live." is cleaner. The impact is obvious without needing to claim free.

**Alternatively**, lean into the warm-up angle for the close: "A vocal warm-up with a professional coach costs $80 and takes a week to schedule. Koda gives you one in 90 seconds." That ties the close directly to what the judges just watched.

---

## 12. Risk Factors

**Ranked by probability x impact:**

1. **Gemini Live flakes during recording (HIGH RISK).** API latency spikes, coaching cues do not fire, greeting takes 8 seconds instead of 2. Mitigation: Record multiple full takes. Use the best moments from each. If Gemini is unresponsive in one take, try again. Have a plan for what to do if Gemini never responds well — you may need to record the demo against a working session and cut around any dead spots.

2. **Strain gauge does not show clear green-to-yellow transition (MEDIUM RISK).** If Daniel sings comfortably and the gauge is already yellow, or if Daniel pushes hard and it stays green, the demo has no contrast. Mitigation: Test with Liza Jane before recording — verify verse/chorus contrast is visible. If contrast is weak, try a different song with clearer easy/hard sections.

3. **Audio mixing is wrong (MEDIUM RISK).** Daniel's singing, Gemini's coaching voice, and the voiceover compete for audio space. If any one is inaudible, the demo fails. Mitigation: Record screen audio and mic audio separately if possible. In editing, follow the mixing rules already in the plan (singing ducks under voiceover, Gemini never ducked, voiceover never overlaps Gemini).

4. **Time crunch leads to a rushed edit (HIGH RISK).** The deadline is today. Recording + editing + voiceover generation + upload is a lot for one day. A rushed edit with bad timing, audio pops, or jarring cuts will make a great product look amateur. Mitigation: Budget 2 hours minimum for editing. If time is short, cut the architecture diagram entirely and use that time for a cleaner edit on the demo footage. A polished 2:15 video with no architecture shot beats a rushed 2:45 video.

5. **GCP deployment fails (MEDIUM RISK).** Cloud Run deployment is "prepped but not yet deployed." If it fails and you cannot fix it in time, you submit without GCP proof and risk disqualification. Mitigation: Deploy to Cloud Run FIRST, before recording. Even if the demo video shows the Fly.io URL, you need a working Cloud Run instance for the submission.

6. **Warm-up angle requires features that are not built (HIGH RISK if attempted).** If you try to implement structured warm-up exercise guidance in Gemini today, you are writing new features on deadline day. Anything that goes wrong with the implementation burns time you need for recording and editing. Mitigation: Reframe, do not rebuild. The warm-up is a narrative wrapper around the existing flow. Gemini's greeting says "let's warm up," the singer starts easy, pushes harder, gets coaching. No new code required — just a tweak to the system prompt and greeting text.

---

## Summary: What to Do Right Now

**Priority order for today:**

1. Deploy to Cloud Run. Non-negotiable. Do this first.
2. Tweak Gemini's greeting to "Let's warm up your voice" (system prompt change, 5 minutes).
3. Test recording: Sing Liza Jane, verify gauge contrast, verify coaching cues fire.
4. Record 3+ full takes. Pick the best moments.
5. Generate voiceover (consider recording yourself if time allows).
6. Edit. Aim for 2:30. Hard cuts. No filler.
7. Upload. Submit. Grab bonus points (blog post, GDG signup).

**If you only have time for one thing from this review:** Fix the "5 parallel analyzers" claim to match reality, and deploy to Cloud Run. Everything else is optimization. Those two are risk-of-disqualification items.

---

## Final Verdict

**Can this win? Yes.** This is a genuinely strong entry. The real-time audio perception, the closed coaching loop, and the Beyond Text Factor are exactly what this competition is designed to reward. The warm-up angle elevates it from "cool tech demo" to "real product." The technical depth is real — not a wrapper.

**Will this win? It depends on execution today.** The gap between "this could win" and "this wins" is: (a) Gemini's coaching responses being genuinely specific and impressive during the recording, (b) the strain gauge showing clear visual contrast, (c) a clean, tight edit that lets the product breathe, and (d) being deployed on Google Cloud.

The product is there. The strategy is there. Now it is about the recording session and the edit. Good luck.
