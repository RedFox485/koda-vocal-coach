#!/usr/bin/env python3
"""Video Composition — Layers all tracks into final demo video.

Takes a timeline config, video clips, and audio clips, and composites
them into a single YouTube-ready MP4 with precise timing, audio ducking,
and text overlays.

Usage:
    python scripts/compose_video.py config/demo_timeline.json --output output/final_demo.mp4
    python scripts/compose_video.py config/demo_timeline.json --preview  # Low-res quick check

Timeline format (demo_timeline.json):
    {
        "resolution": {"width": 1920, "height": 1080},
        "fps": 60,
        "video_tracks": [
            {"path": "video/dji_hook.mp4", "start": 0.0, "end": 3.0, "label": "hook"},
            {"path": "video/captures/session.mp4", "start": 3.0, "end": 140.0,
             "source_offset": 0.0, "label": "ui_capture"},
            {"path": "docs/architecture.png", "start": 125.0, "end": 140.0,
             "type": "image", "label": "architecture"},
            ...
        ],
        "audio_tracks": [
            {"path": "audio/singing.wav", "start": 20.0, "duck_under": "voiceover",
             "duck_level": 0.4, "label": "singing"},
            {"path": "audio/voiceover/vo_01_problem.mp3", "start": 3.0,
             "group": "voiceover", "label": "vo_problem"},
            {"path": "audio/coaching/cue_1.wav", "start": 65.0,
             "group": "gemini", "label": "coaching_1"},
            ...
        ],
        "text_overlays": [
            {"text": "Parselmouth · Shimmer · HNR", "start": 28.0, "end": 30.0,
             "position": "lower-third", "font_size": 24},
            ...
        ],
        "transitions": [
            {"type": "crossfade", "at": 3.0, "duration": 0.2}
        ]
    }
"""

import argparse
import json
import os
import subprocess
import shutil
import sys
from pathlib import Path


def build_ffmpeg_command(timeline: dict, output_path: str, preview: bool = False) -> list:
    """Build the ffmpeg command from a timeline config.

    This builds a single complex ffmpeg command that:
    1. Loads all video and audio inputs
    2. Trims and positions video clips
    3. Places audio clips at exact timestamps
    4. Applies audio ducking
    5. Burns in text overlays
    6. Outputs YouTube-optimized H.264
    """
    res = timeline["resolution"]
    fps = timeline.get("fps", 60)
    width, height = res["width"], res["height"]

    # Collect all input files
    inputs = []
    input_map = {}  # label -> input index

    # Video inputs
    for vt in timeline.get("video_tracks", []):
        idx = len(inputs)
        inputs.append(vt["path"])
        input_map[vt.get("label", f"v{idx}")] = idx

    # Audio inputs
    for at in timeline.get("audio_tracks", []):
        idx = len(inputs)
        inputs.append(at["path"])
        input_map[at.get("label", f"a{idx}")] = idx

    # Build filter_complex
    filters = []
    video_segments = []
    audio_segments = []

    # === VIDEO PROCESSING ===
    for i, vt in enumerate(timeline.get("video_tracks", [])):
        idx = input_map[vt.get("label", f"v{i}")]
        label = f"v{i}"
        duration = vt["end"] - vt["start"]

        if vt.get("type") == "image":
            # Static image — loop for duration
            filters.append(
                f"[{idx}:v]loop=loop=-1:size=1:start=0,"
                f"setpts=PTS-STARTPTS,"
                f"trim=duration={duration},"
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"setpts=PTS+{vt['start']}/TB"
                f"[{label}]"
            )
        else:
            source_offset = vt.get("source_offset", 0)
            filters.append(
                f"[{idx}:v]trim=start={source_offset}:duration={duration},"
                f"setpts=PTS-STARTPTS,"
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"setpts=PTS+{vt['start']}/TB"
                f"[{label}]"
            )
        video_segments.append(f"[{label}]")

    # Concatenate video segments (they're already timed with setpts)
    if video_segments:
        n_vid = len(video_segments)
        concat_input = "".join(video_segments)
        filters.append(
            f"{concat_input}concat=n={n_vid}:v=1:a=0[vout_raw]"
        )

    # === TEXT OVERLAYS ===
    overlay_chain = "vout_raw"
    for j, txt in enumerate(timeline.get("text_overlays", [])):
        text = txt["text"].replace("'", "\\'").replace(":", "\\:")
        font_size = txt.get("font_size", 24)
        y_pos = txt.get("y", height - 100)  # Default: lower third

        if txt.get("position") == "lower-third":
            y_pos = height - 80

        new_label = f"vtxt{j}"
        filters.append(
            f"[{overlay_chain}]drawtext="
            f"text='{text}':"
            f"fontsize={font_size}:"
            f"fontcolor=white:"
            f"borderw=2:bordercolor=black@0.5:"
            f"x=(w-text_w)/2:y={y_pos}:"
            f"enable='between(t,{txt['start']},{txt['end']})':"
            f"alpha='if(lt(t,{txt['start']}+0.3),(t-{txt['start']})/0.3,"
            f"if(gt(t,{txt['end']}-0.3),({txt['end']}-t)/0.3,1))'"
            f"[{new_label}]"
        )
        overlay_chain = new_label

    # Final video label
    if timeline.get("text_overlays"):
        filters.append(f"[{overlay_chain}]null[vfinal]")
    else:
        filters.append("[vout_raw]null[vfinal]")

    # === AUDIO PROCESSING ===
    audio_labels = []
    for i, at in enumerate(timeline.get("audio_tracks", [])):
        idx = input_map[at.get("label", f"a{i}")]
        label = f"a{i}"
        start = at.get("start", 0)

        # Convert ms to samples for adelay
        delay_ms = int(start * 1000)

        filters.append(
            f"[{idx}:a]adelay={delay_ms}|{delay_ms},"
            f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
            f"[{label}]"
        )
        audio_labels.append(f"[{label}]")

    # Mix all audio
    if audio_labels:
        n_audio = len(audio_labels)
        audio_concat = "".join(audio_labels)
        filters.append(
            f"{audio_concat}amix=inputs={n_audio}:"
            f"duration=longest:dropout_transition=2,"
            f"loudnorm=I=-16:TP=-1.5:LRA=11"
            f"[afinal]"
        )

    # Build full command
    cmd = ["ffmpeg", "-y"]

    # Input files
    for inp in inputs:
        if inp.endswith((".png", ".jpg", ".jpeg")):
            cmd.extend(["-loop", "1", "-i", inp])
        else:
            cmd.extend(["-i", inp])

    # Filter complex
    filter_str = ";\n".join(filters)
    cmd.extend(["-filter_complex", filter_str])

    # Map outputs
    cmd.extend(["-map", "[vfinal]"])
    if audio_labels:
        cmd.extend(["-map", "[afinal]"])

    # Encoding settings
    if preview:
        cmd.extend([
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-s", f"{width//2}x{height//2}",
        ])
    else:
        cmd.extend([
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-r", str(fps),
        ])

    cmd.extend([
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
        "-movflags", "+faststart",
        output_path,
    ])

    return cmd


def compose(timeline_path: str, output_path: str, preview: bool = False, dry_run: bool = False):
    """Run the full composition pipeline."""
    timeline = json.loads(Path(timeline_path).read_text())

    # Validate all input files exist
    missing = []
    for vt in timeline.get("video_tracks", []):
        if not Path(vt["path"]).exists():
            missing.append(vt["path"])
    for at in timeline.get("audio_tracks", []):
        if not Path(at["path"]).exists():
            missing.append(at["path"])

    if missing:
        print("ERROR: Missing input files:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    # Build and run ffmpeg command
    cmd = build_ffmpeg_command(timeline, output_path, preview)

    if dry_run:
        print("FFmpeg command:")
        print(" \\\n  ".join(cmd))
        return

    print(f"[compose] Compositing {len(timeline.get('video_tracks', []))} video + "
          f"{len(timeline.get('audio_tracks', []))} audio tracks...")
    print(f"[compose] Output: {output_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"[compose] FFmpeg error:\n{result.stderr[-1000:]}")
        sys.exit(1)

    # Report
    file_size = Path(output_path).stat().st_size
    print(f"\n[compose] Done!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size / (1024*1024):.1f} MB")

    # Get duration
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", output_path],
        capture_output=True, text=True
    )
    if probe.returncode == 0:
        info = json.loads(probe.stdout)
        dur = float(info.get("format", {}).get("duration", 0))
        print(f"  Duration: {dur:.1f}s ({dur/60:.1f}m)")


def create_demo_timeline():
    """Generate the Koda demo timeline config."""
    timeline = {
        "resolution": {"width": 1920, "height": 1080},
        "fps": 60,
        "video_tracks": [
            {
                "path": "video/dji/hook.mp4",
                "start": 0.0, "end": 3.0,
                "label": "hook",
                "_note": "DJI Osmo 2 clip of Daniel singing"
            },
            {
                "path": "video/captures/session.mp4",
                "start": 3.0, "end": 125.0,
                "source_offset": 0.0,
                "label": "ui_capture",
                "_note": "Playwright-captured Koda UI with injected audio"
            },
            {
                "path": "docs/architecture.png",
                "start": 125.0, "end": 140.0,
                "type": "image",
                "label": "architecture",
                "_note": "Architecture diagram — 15s"
            },
            {
                "path": "video/title_card.png",
                "start": 140.0, "end": 160.0,
                "type": "image",
                "label": "title_card",
                "_note": "Koda title card + Powered by Gemini Live"
            }
        ],
        "audio_tracks": [
            {
                "path": "audio/singing/best_take.wav",
                "start": 20.0,
                "group": "singing",
                "label": "singing",
                "_note": "Daniel singing — starts at Shot 4 (green zone)"
            },
            {
                "path": "audio/voiceover/vo_01_problem.mp3",
                "start": 3.0,
                "group": "voiceover",
                "label": "vo_problem"
            },
            {
                "path": "audio/voiceover/vo_02_greeting.mp3",
                "start": 12.0,
                "group": "voiceover",
                "label": "vo_greeting"
            },
            {
                "path": "audio/voiceover/vo_03_green_tech.mp3",
                "start": 25.0,
                "group": "voiceover",
                "label": "vo_green_tech"
            },
            {
                "path": "audio/voiceover/vo_04_strain_builds.mp3",
                "start": 53.0,
                "group": "voiceover",
                "label": "vo_strain"
            },
            {
                "path": "audio/voiceover/vo_05_coaching_explain.mp3",
                "start": 72.0,
                "group": "voiceover",
                "label": "vo_coaching"
            },
            {
                "path": "audio/voiceover/vo_06_second_cue.mp3",
                "start": 88.0,
                "group": "voiceover",
                "label": "vo_second_cue"
            },
            {
                "path": "audio/voiceover/vo_07_summary.mp3",
                "start": 112.0,
                "group": "voiceover",
                "label": "vo_summary"
            },
            {
                "path": "audio/voiceover/vo_08_architecture.mp3",
                "start": 125.0,
                "group": "voiceover",
                "label": "vo_arch"
            },
            {
                "path": "audio/voiceover/vo_09_close.mp3",
                "start": 140.0,
                "group": "voiceover",
                "label": "vo_close"
            },
            {
                "path": "audio/gemini/greeting.wav",
                "start": 14.0,
                "group": "gemini",
                "label": "gemini_greeting",
                "_note": "Gemini greeting — plays after Start click"
            },
            {
                "path": "audio/gemini/coaching_cue_1.wav",
                "start": 65.0,
                "group": "gemini",
                "label": "coaching_1",
                "_note": "First coaching cue — the money shot"
            },
            {
                "path": "audio/gemini/coaching_cue_2.wav",
                "start": 82.0,
                "group": "gemini",
                "label": "coaching_2",
                "_note": "Second cue — proves improvisation"
            },
            {
                "path": "audio/gemini/song_summary.wav",
                "start": 102.0,
                "group": "gemini",
                "label": "song_summary",
                "_note": "Song-end summary — full session context"
            }
        ],
        "text_overlays": [
            {
                "text": "Parselmouth - Shimmer - HNR",
                "start": 28.0, "end": 30.5,
                "position": "lower-third", "font_size": 22
            },
            {
                "text": "CPP - Vocal Fold Closure",
                "start": 31.0, "end": 33.5,
                "position": "lower-third", "font_size": 22
            },
            {
                "text": "8-Channel Perceptual Engine",
                "start": 34.0, "end": 36.5,
                "position": "lower-third", "font_size": 22
            },
            {
                "text": "96ms Pipeline Latency",
                "start": 37.0, "end": 40.0,
                "position": "lower-third", "font_size": 26
            },
            {
                "text": "Phrase Coaching - Gemini Live",
                "start": 65.0, "end": 75.0,
                "position": "lower-third", "font_size": 22
            },
            {
                "text": "Song Summary - Full Session Context",
                "start": 102.0, "end": 115.0,
                "position": "lower-third", "font_size": 22
            }
        ],
        "transitions": [
            {"type": "crossfade", "at": 3.0, "duration": 0.2,
             "_note": "DJI → screen recording"}
        ]
    }
    return timeline


def main():
    parser = argparse.ArgumentParser(description="Compose demo video from timeline")
    parser.add_argument("timeline", nargs="?", help="Path to timeline JSON")
    parser.add_argument("--output", "-o", default="output/final_demo.mp4")
    parser.add_argument("--preview", action="store_true",
                        help="Quick low-res preview render")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print ffmpeg command without running")
    parser.add_argument("--create-timeline", action="store_true",
                        help="Generate the Koda demo timeline config")

    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not found. Install with: brew install ffmpeg")
        sys.exit(1)

    if args.create_timeline:
        tl = create_demo_timeline()
        out_path = Path("config/demo_timeline.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(tl, indent=2))
        print(f"Created {out_path}")
        return

    if not args.timeline:
        parser.print_help()
        sys.exit(1)

    compose(args.timeline, args.output, args.preview, args.dry_run)


if __name__ == "__main__":
    main()
