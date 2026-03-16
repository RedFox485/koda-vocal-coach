#!/usr/bin/env python3
"""Demo Video Renderer — Autonomous production pipeline.

One command → complete, polished demo video.

Usage:
    # Full pipeline (requires all inputs + ElevenLabs API key)
    python scripts/render_demo.py \
        --singing audio/singing/best_take.wav \
        --dji video/dji/hook.mp4 \
        --output output/final_demo.mp4

    # Skip voiceover generation (use existing clips)
    python scripts/render_demo.py \
        --singing audio/singing/best_take.wav \
        --skip-voiceover \
        --output output/final_demo.mp4

    # Generate config files only (for manual editing)
    python scripts/render_demo.py --init

    # Dry run (show what would happen)
    python scripts/render_demo.py --singing audio/test.wav --dry-run

Pipeline:
    1. Parse inputs and build timeline
    2. Generate voiceover clips (ElevenLabs API)
    3. Launch Koda backend
    4. Inject singing audio via WebSocket + capture UI via Playwright
    5. Compose all tracks into final video (FFmpeg)
    6. QA check (extract keyframes, verify)
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Pipeline scripts (imported as needed)
PROJECT_ROOT = Path(__file__).parent.parent


def init_project():
    """Create directory structure and config templates."""
    dirs = [
        "audio/singing",
        "audio/voiceover",
        "audio/gemini",
        "video/dji",
        "video/captures",
        "video/frames",
        "output",
        "config",
    ]
    for d in dirs:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {d}/")

    # Generate script config
    from generate_voiceover import create_demo_script
    script = create_demo_script()
    script_path = PROJECT_ROOT / "config" / "demo_script.json"
    script_path.write_text(json.dumps(script, indent=2))
    print(f"  Created: config/demo_script.json")

    # Generate timeline config
    from compose_video import create_demo_timeline
    timeline = create_demo_timeline()
    timeline_path = PROJECT_ROOT / "config" / "demo_timeline.json"
    timeline_path.write_text(json.dumps(timeline, indent=2))
    print(f"  Created: config/demo_timeline.json")

    # Generate pauses config
    pauses = [
        {"at": 45.0, "duration": 8.0, "label": "coaching_cue_1"},
        {"at": 62.0, "duration": 8.0, "label": "coaching_cue_2"},
        {"at": 80.0, "duration": 16.0, "label": "song_end_silence"},
    ]
    pauses_path = PROJECT_ROOT / "config" / "audio_pauses.json"
    pauses_path.write_text(json.dumps(pauses, indent=2))
    print(f"  Created: config/audio_pauses.json")

    print(f"\nNext steps:")
    print(f"  1. Place singing audio in audio/singing/")
    print(f"  2. Place DJI clip in video/dji/")
    print(f"  3. Set ELEVENLABS_API_KEY and edit voice_id in config/demo_script.json")
    print(f"  4. Edit config/audio_pauses.json with actual phrase boundary timestamps")
    print(f"  5. Run: python scripts/render_demo.py --singing audio/singing/take.wav")


async def run_pipeline(singing_path: str, dji_path: str = None,
                       output_path: str = "output/final_demo.mp4",
                       skip_voiceover: bool = False,
                       skip_capture: bool = False,
                       backend_url: str = "http://localhost:8000",
                       dry_run: bool = False):
    """Execute the full render pipeline."""

    ws_url = backend_url.replace("http", "ws") + "/ws"
    timings = {}

    print("=" * 60)
    print("KODA DEMO VIDEO RENDERER")
    print("=" * 60)
    print(f"  Singing: {singing_path}")
    print(f"  DJI clip: {dji_path or 'none'}")
    print(f"  Output: {output_path}")
    print(f"  Backend: {backend_url}")
    print()

    # ── PHASE 1: VOICEOVER GENERATION ──────────────────────────

    if not skip_voiceover:
        print("\n── PHASE 1: Voiceover Generation ──")
        t0 = time.monotonic()

        script_path = PROJECT_ROOT / "config" / "demo_script.json"
        if not script_path.exists():
            print("ERROR: config/demo_script.json not found. Run --init first.")
            sys.exit(1)

        vo_dir = PROJECT_ROOT / "audio" / "voiceover"
        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "generate_voiceover.py"),
            str(script_path), "--output", str(vo_dir),
        ]

        if dry_run:
            print(f"  Would run: {' '.join(cmd)}")
        else:
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if result.returncode != 0:
                print("ERROR: Voiceover generation failed")
                sys.exit(1)

        timings["voiceover"] = time.monotonic() - t0
        print(f"  Voiceover done in {timings['voiceover']:.1f}s")
    else:
        print("\n── PHASE 1: Skipped (--skip-voiceover) ──")

    # ── PHASE 2: UI CAPTURE ────────────────────────────────────

    if not skip_capture:
        print("\n── PHASE 2: UI Capture ──")
        t0 = time.monotonic()

        # Check backend is running
        if not dry_run:
            try:
                import urllib.request
                urllib.request.urlopen(backend_url, timeout=3)
                print(f"  Backend is running at {backend_url}")
            except Exception:
                print(f"  WARNING: Backend not reachable at {backend_url}")
                print(f"  Start it with: cd {PROJECT_ROOT} && python -m uvicorn src.vocal_health_backend:app --port 8000")
                print(f"  Then re-run this script with --skip-voiceover")
                sys.exit(1)

        # Run audio injection and UI capture in parallel
        capture_path = PROJECT_ROOT / "video" / "captures" / "session.mp4"
        pauses_path = PROJECT_ROOT / "config" / "audio_pauses.json"

        # Determine duration from audio file
        import soundfile as sf
        audio_data, sr = sf.read(singing_path, dtype='float32')
        audio_duration = len(audio_data) / sr

        # Add time for: Gemini greeting (~8s) + post-song silence (~20s)
        total_capture_duration = 8 + audio_duration + 20

        if dry_run:
            print(f"  Would capture {total_capture_duration:.0f}s of UI")
            print(f"  Would inject {audio_duration:.0f}s of audio")
        else:
            # Launch capture in background
            capture_cmd = [
                sys.executable, str(PROJECT_ROOT / "scripts" / "capture_ui.py"),
                "--url", backend_url,
                "--output", str(capture_path),
                "--mode", "hybrid",
                "--duration", str(total_capture_duration),
                "--auto-start",
            ]
            capture_proc = subprocess.Popen(capture_cmd, cwd=str(PROJECT_ROOT))

            # Wait for page load + Start click + Gemini greeting
            await asyncio.sleep(8)

            # Inject audio
            pauses = None
            if pauses_path.exists():
                pauses = json.loads(pauses_path.read_text())

            from inject_audio import inject
            await inject(ws_url, singing_path, pauses=pauses)

            # Wait for capture to finish
            capture_proc.wait(timeout=60)

            if capture_proc.returncode != 0:
                print("WARNING: Capture may have had issues")

        timings["capture"] = time.monotonic() - t0
        print(f"  Capture done in {timings['capture']:.1f}s")
    else:
        print("\n── PHASE 2: Skipped (--skip-capture) ──")

    # ── PHASE 3: COMPOSITION ───────────────────────────────────

    print("\n── PHASE 3: Video Composition ──")
    t0 = time.monotonic()

    timeline_path = PROJECT_ROOT / "config" / "demo_timeline.json"
    if not timeline_path.exists():
        print("ERROR: config/demo_timeline.json not found. Run --init first.")
        sys.exit(1)

    # Update timeline with actual paths
    timeline = json.loads(timeline_path.read_text())

    # Inject DJI path if provided
    if dji_path:
        for vt in timeline["video_tracks"]:
            if vt.get("label") == "hook":
                vt["path"] = dji_path

    # Save updated timeline
    runtime_timeline = PROJECT_ROOT / "config" / "demo_timeline_runtime.json"
    runtime_timeline.write_text(json.dumps(timeline, indent=2))

    compose_cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "compose_video.py"),
        str(runtime_timeline), "--output", output_path,
    ]

    if dry_run:
        compose_cmd.append("--dry-run")

    result = subprocess.run(compose_cmd, cwd=str(PROJECT_ROOT))

    timings["compose"] = time.monotonic() - t0
    print(f"  Composition done in {timings['compose']:.1f}s")

    # ── PHASE 4: QA CHECK ──────────────────────────────────────

    if not dry_run and Path(output_path).exists():
        print("\n── PHASE 4: QA Check ──")

        # Extract keyframes at shot boundaries
        shot_times = [0, 3, 12, 20, 48, 65, 82, 98, 125, 140]
        qa_dir = PROJECT_ROOT / "output" / "qa_frames"
        qa_dir.mkdir(parents=True, exist_ok=True)

        for t in shot_times:
            frame_path = qa_dir / f"frame_{t:03d}s.png"
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(t), "-i", output_path,
                "-frames:v", "1", "-q:v", "2", str(frame_path),
            ], capture_output=True)

        print(f"  Extracted {len(shot_times)} QA frames to {qa_dir}/")

        # Get final video info
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", output_path],
            capture_output=True, text=True
        )
        if probe.returncode == 0:
            info = json.loads(probe.stdout)
            fmt = info.get("format", {})
            duration = float(fmt.get("duration", 0))
            size_mb = int(fmt.get("size", 0)) / (1024 * 1024)
            print(f"  Duration: {duration:.1f}s ({duration/60:.1f}m)")
            print(f"  File size: {size_mb:.1f} MB")

            for stream in info.get("streams", []):
                if stream["codec_type"] == "video":
                    print(f"  Video: {stream.get('width')}x{stream.get('height')} "
                          f"{stream.get('codec_name')} @ {stream.get('r_frame_rate')} fps")
                elif stream["codec_type"] == "audio":
                    print(f"  Audio: {stream.get('codec_name')} "
                          f"{stream.get('sample_rate')}Hz "
                          f"{stream.get('bit_rate', 'N/A')} bps")

    # ── SUMMARY ────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("RENDER COMPLETE")
    print("=" * 60)
    total = sum(timings.values())
    for phase, dur in timings.items():
        print(f"  {phase}: {dur:.1f}s")
    print(f"  TOTAL: {total:.1f}s ({total/60:.1f}m)")
    if not dry_run:
        print(f"\n  Output: {output_path}")
        print(f"  QA frames: output/qa_frames/")


def main():
    parser = argparse.ArgumentParser(
        description="Koda Demo Video Renderer — Autonomous production pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Initialize project:     python scripts/render_demo.py --init
  Full render:            python scripts/render_demo.py --singing audio/singing/take.wav
  Skip voiceover:         python scripts/render_demo.py --singing audio/singing/take.wav --skip-voiceover
  Dry run:                python scripts/render_demo.py --singing audio/singing/take.wav --dry-run
        """,
    )
    parser.add_argument("--singing", help="Path to singing audio file")
    parser.add_argument("--dji", help="Path to DJI Osmo 2 hook clip")
    parser.add_argument("--output", "-o", default="output/final_demo.mp4")
    parser.add_argument("--backend", default="http://localhost:8000",
                        help="Koda backend URL")
    parser.add_argument("--skip-voiceover", action="store_true")
    parser.add_argument("--skip-capture", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--init", action="store_true",
                        help="Create directory structure and config templates")

    args = parser.parse_args()

    if args.init:
        # Need to be in project root for relative imports
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        os.chdir(str(PROJECT_ROOT))
        init_project()
        return

    if not args.singing and not args.skip_capture:
        print("ERROR: --singing is required (or use --skip-capture)")
        parser.print_help()
        sys.exit(1)

    asyncio.run(run_pipeline(
        singing_path=args.singing,
        dji_path=args.dji,
        output_path=args.output,
        skip_voiceover=args.skip_voiceover,
        skip_capture=args.skip_capture,
        backend_url=args.backend,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
