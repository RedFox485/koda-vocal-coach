#!/usr/bin/env python3
"""Generate voiceover clips from script using ElevenLabs API.

Usage:
    python scripts/generate_voiceover.py config/demo_script.json --output audio/voiceover/
    python scripts/generate_voiceover.py config/demo_script.json --voice-id <id> --output audio/voiceover/
    python scripts/generate_voiceover.py --list-voices

Script format (demo_script.json):
    {
        "voice_id": "optional_override",
        "model": "eleven_multilingual_v2",
        "clips": [
            {
                "id": "vo_01_problem",
                "text": "Hundreds of millions of people sing without a teacher...",
                "stability": 0.65,
                "clarity": 0.75,
                "style": 0.3,
                "speed": 0.95
            },
            ...
        ]
    }

Environment:
    ELEVENLABS_API_KEY — your API key (or pass --api-key)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from elevenlabs import ElevenLabs
    HAS_SDK = True
except ImportError:
    HAS_SDK = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# Default voice settings for "calm, confident engineering narration"
DEFAULTS = {
    "model": "eleven_multilingual_v2",
    "stability": 0.65,        # Slight natural variation
    "similarity_boost": 0.75, # Balance between clone accuracy and naturalness
    "style": 0.3,             # Subtle warmth
    "speed": 1.0,             # Normal speed (adjust per-clip)
}


def generate_with_sdk(api_key: str, voice_id: str, clips: list, model: str,
                      output_dir: Path) -> list:
    """Generate clips using the official ElevenLabs Python SDK."""
    client = ElevenLabs(api_key=api_key)

    results = []
    for i, clip in enumerate(clips):
        clip_id = clip.get("id", f"vo_{i+1:02d}")
        text = clip["text"]
        output_path = output_dir / f"{clip_id}.mp3"

        print(f"[{i+1}/{len(clips)}] Generating: {clip_id} ({len(text)} chars)")
        print(f"  Text: {text[:80]}{'...' if len(text) > 80 else ''}")

        try:
            audio = client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=clip.get("model", model),
                voice_settings={
                    "stability": clip.get("stability", DEFAULTS["stability"]),
                    "similarity_boost": clip.get("clarity", DEFAULTS["similarity_boost"]),
                    "style": clip.get("style", DEFAULTS["style"]),
                    "use_speaker_boost": True,
                },
            )

            # SDK returns a generator, write bytes
            with open(output_path, "wb") as f:
                for chunk in audio:
                    f.write(chunk)

            file_size = output_path.stat().st_size
            print(f"  Saved: {output_path} ({file_size / 1024:.1f} KB)")

            results.append({
                "id": clip_id,
                "path": str(output_path),
                "size_bytes": file_size,
                "text": text,
                "char_count": len(text),
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"id": clip_id, "error": str(e)})

        # Rate limit courtesy
        if i < len(clips) - 1:
            time.sleep(0.5)

    return results


def generate_with_requests(api_key: str, voice_id: str, clips: list, model: str,
                           output_dir: Path) -> list:
    """Generate clips using raw HTTP requests (no SDK needed)."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    results = []
    for i, clip in enumerate(clips):
        clip_id = clip.get("id", f"vo_{i+1:02d}")
        text = clip["text"]
        output_path = output_dir / f"{clip_id}.mp3"

        print(f"[{i+1}/{len(clips)}] Generating: {clip_id} ({len(text)} chars)")

        payload = {
            "text": text,
            "model_id": clip.get("model", model),
            "voice_settings": {
                "stability": clip.get("stability", DEFAULTS["stability"]),
                "similarity_boost": clip.get("clarity", DEFAULTS["similarity_boost"]),
                "style": clip.get("style", DEFAULTS["style"]),
                "use_speaker_boost": True,
            },
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(resp.content)

            file_size = output_path.stat().st_size
            print(f"  Saved: {output_path} ({file_size / 1024:.1f} KB)")
            results.append({"id": clip_id, "path": str(output_path), "size_bytes": file_size})

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"id": clip_id, "error": str(e)})

        if i < len(clips) - 1:
            time.sleep(0.5)

    return results


def list_voices(api_key: str):
    """Print available voices."""
    if HAS_SDK:
        client = ElevenLabs(api_key=api_key)
        voices = client.voices.get_all()
        for v in voices.voices:
            print(f"  {v.voice_id}  {v.name}  ({', '.join(v.labels.values()) if v.labels else 'no labels'})")
    elif HAS_REQUESTS:
        resp = requests.get("https://api.elevenlabs.io/v1/voices",
                            headers={"xi-api-key": api_key})
        for v in resp.json().get("voices", []):
            labels = ", ".join(v.get("labels", {}).values())
            print(f"  {v['voice_id']}  {v['name']}  ({labels})")


def create_demo_script():
    """Generate the Koda demo script file from the production plan."""
    script = {
        "voice_id": "REPLACE_WITH_YOUR_VOICE_ID",
        "model": "eleven_multilingual_v2",
        "clips": [
            {
                "id": "vo_01_problem",
                "text": "Hundreds of millions of people sing without a teacher. Nobody tells them when they're hurting their voice. This is Koda.",
                "stability": 0.70,
                "style": 0.2,
                "speed": 0.95,
                "_note": "Low energy. Matter-of-fact. 'This is Koda' gets a slight pause before it."
            },
            {
                "id": "vo_02_greeting",
                "text": "One tap. Gemini introduces itself.",
                "stability": 0.65,
                "style": 0.2,
                "speed": 1.0,
                "_note": "Quick and clean. Throwaway — Gemini's voice is the star here."
            },
            {
                "id": "vo_03_green_tech",
                "text": "Five acoustic analyzers run in parallel on every audio frame. Parselmouth extracts shimmer and harmonic-to-noise ratio. A cepstral peak prominence detector measures vocal fold closure. An eight-channel perceptual engine scores timbral strain. Total pipeline latency: ninety-six milliseconds. Green means healthy phonation — the baseline adapts to your voice, not fixed thresholds.",
                "stability": 0.60,
                "style": 0.3,
                "speed": 0.95,
                "_note": "Steady engineering confidence. Each analyzer name gets slight emphasis. 'Ninety-six milliseconds' = understated pride."
            },
            {
                "id": "vo_04_strain_builds",
                "text": "When technique breaks down, Koda sees it immediately. Shimmer spikes as vocal fold vibration becomes irregular. Cepstral peak prominence drops. The strain engine fuses these signals in real-time.",
                "stability": 0.55,
                "style": 0.4,
                "speed": 1.0,
                "_note": "Energy rises with the strain. Engaged — 'watch this' energy."
            },
            {
                "id": "vo_05_coaching_explain",
                "text": "At every breath point, Gemini speaks a technique cue. Not scripted — it reads the strain score, phrase duration, and vocal mode, then improvises.",
                "stability": 0.60,
                "style": 0.35,
                "speed": 0.95,
                "_note": "After coaching cue plays. Narrator sounds genuinely impressed."
            },
            {
                "id": "vo_06_second_cue",
                "text": "Every cue is different. Gemini improvises within technique-anchored style guidelines, using the last five phrases of context.",
                "stability": 0.60,
                "style": 0.35,
                "speed": 0.95,
                "_note": "After second coaching cue. Real appreciation, not fake surprise."
            },
            {
                "id": "vo_07_summary",
                "text": "After each song, Gemini delivers a spoken summary of the entire session. One persistent connection — every phrase, every strain spike, every zone transition stays in context. No resets.",
                "stability": 0.65,
                "style": 0.3,
                "speed": 0.90,
                "_note": "Warm. 'No resets' is the technical kicker — land it cleanly."
            },
            {
                "id": "vo_08_architecture",
                "text": "Browser captures audio over WebSocket at ten hertz. Cloud Run backend runs five analyzers in parallel — Parselmouth, C P P, perceptual engine, wavelet scattering, phonation classifier. The strain engine fuses signals and triggers Gemini Live at each phrase boundary. Native audio generation — not T T S.",
                "stability": 0.55,
                "style": 0.2,
                "speed": 1.10,
                "_note": "Crisp, FAST. Rapid-fire technical. 'Not TTS' gets a pause before it."
            },
            {
                "id": "vo_09_close",
                "text": "A voice lesson costs eighty dollars an hour — if you can find a teacher at all. Koda gives every singer a real-time vocal health coach, anywhere, in any language, for free. Powered by Gemini Live.",
                "stability": 0.70,
                "style": 0.45,
                "speed": 0.85,
                "_note": "SLOW. Emotional peak. Pause before 'for free'. 'Powered by Gemini Live' = pride, not hype."
            },
        ]
    }
    return script


def main():
    parser = argparse.ArgumentParser(description="Generate voiceover clips via ElevenLabs")
    parser.add_argument("script", nargs="?", help="Path to script JSON file")
    parser.add_argument("--output", "-o", default="audio/voiceover/",
                        help="Output directory for audio clips")
    parser.add_argument("--voice-id", help="Override voice ID from script")
    parser.add_argument("--api-key", help="ElevenLabs API key (or set ELEVENLABS_API_KEY)")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")
    parser.add_argument("--create-script", action="store_true",
                        help="Generate the Koda demo script JSON file")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ELEVENLABS_API_KEY")

    if args.create_script:
        script = create_demo_script()
        out_path = Path("config/demo_script.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(script, indent=2))
        print(f"Created {out_path}")
        print(f"Edit voice_id, then run: python scripts/generate_voiceover.py {out_path}")
        return

    if args.list_voices:
        if not api_key:
            print("ERROR: Set ELEVENLABS_API_KEY or pass --api-key")
            sys.exit(1)
        list_voices(api_key)
        return

    if not args.script:
        parser.print_help()
        sys.exit(1)

    if not api_key:
        print("ERROR: Set ELEVENLABS_API_KEY or pass --api-key")
        sys.exit(1)

    # Load script
    script = json.loads(Path(args.script).read_text())
    clips = script["clips"]
    voice_id = args.voice_id or script.get("voice_id", "")
    model = script.get("model", DEFAULTS["model"])

    if not voice_id or voice_id == "REPLACE_WITH_YOUR_VOICE_ID":
        print("ERROR: Set voice_id in script JSON or pass --voice-id")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(clips)} clips with voice {voice_id}")
    print(f"Output: {output_dir}/\n")

    # Generate
    if HAS_SDK:
        results = generate_with_sdk(api_key, voice_id, clips, model, output_dir)
    elif HAS_REQUESTS:
        print("(Using requests — install elevenlabs SDK for best results)")
        results = generate_with_requests(api_key, voice_id, clips, model, output_dir)
    else:
        print("ERROR: Install either 'elevenlabs' or 'requests' package")
        sys.exit(1)

    # Summary
    print(f"\n{'='*60}")
    success = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    total_chars = sum(r.get("char_count", 0) for r in success)
    print(f"Generated: {len(success)}/{len(clips)} clips")
    print(f"Total characters: {total_chars}")
    if failed:
        print(f"Failed: {[r['id'] for r in failed]}")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
