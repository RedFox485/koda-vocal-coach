#!/usr/bin/env python3
"""Upload a video to YouTube using the YouTube Data API v3.

Usage:
    python scripts/youtube_upload.py video/output/koda_demo_v4.mp4 \
        --title "Koda — Real-Time Vocal Coach Powered by Gemini Live" \
        --description-file docs/youtube-metadata.md \
        --tags "Gemini Live,Gemini API,Google Cloud Run,vocal coach,AI" \
        --category 28 \
        --privacy public

First run will open a browser for OAuth consent. Token is cached for future uploads.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# YouTube upload scope
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# Default paths
ROOT = Path(__file__).parent.parent
TOKEN_FILE = ROOT / ".youtube_token.json"
CLIENT_SECRETS = ROOT / "client_secret.json"


def get_credentials(client_secrets_path: str = None) -> Credentials:
    """Get or refresh OAuth credentials."""
    creds = None
    secrets = Path(client_secrets_path) if client_secrets_path else CLIENT_SECRETS

    # Load cached token
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    # Refresh or get new token
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("[youtube] Refreshing token...")
            creds.refresh(Request())
        else:
            if not secrets.exists():
                print(f"[youtube] ERROR: Client secrets not found at {secrets}")
                print("[youtube] Download from Google Cloud Console → APIs → Credentials → OAuth client → Download JSON")
                sys.exit(1)
            print("[youtube] Opening browser for OAuth consent...")
            flow = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES)
            creds = flow.run_local_server(port=8090)

        # Save token for future use
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
        print(f"[youtube] Token saved to {TOKEN_FILE}")

    return creds


def upload_video(
    video_path: str,
    title: str,
    description: str = "",
    tags: list = None,
    category_id: str = "28",  # 28 = Science & Technology
    privacy: str = "public",
    client_secrets: str = None,
) -> str:
    """Upload a video to YouTube. Returns the video URL."""
    creds = get_credentials(client_secrets)
    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags or [],
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        video_path,
        mimetype="video/mp4",
        resumable=True,
        chunksize=10 * 1024 * 1024,  # 10MB chunks
    )

    print(f"[youtube] Uploading {Path(video_path).name} ({Path(video_path).stat().st_size / 1024 / 1024:.1f} MB)...")
    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"[youtube] Upload {int(status.progress() * 100)}%")

    video_id = response["id"]
    video_url = f"https://youtube.com/watch?v={video_id}"
    print(f"[youtube] Upload complete!")
    print(f"[youtube] URL: {video_url}")
    return video_url


def parse_description_file(path: str) -> str:
    """Extract description from youtube-metadata.md."""
    content = Path(path).read_text()
    # Find the ## Description section
    in_desc = False
    lines = []
    for line in content.split("\n"):
        if line.startswith("## Description"):
            in_desc = True
            continue
        if in_desc and line.startswith("## "):
            break
        if in_desc:
            lines.append(line)
    return "\n".join(lines).strip() if lines else content


def main():
    parser = argparse.ArgumentParser(description="Upload video to YouTube")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--title", required=True, help="Video title")
    parser.add_argument("--description", default="", help="Video description")
    parser.add_argument("--description-file", help="Read description from file (markdown)")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--category", default="28", help="YouTube category ID (28=Science & Technology)")
    parser.add_argument("--privacy", default="public", choices=["public", "unlisted", "private"])
    parser.add_argument("--client-secrets", help="Path to OAuth client secrets JSON")
    args = parser.parse_args()

    # Get description
    description = args.description
    if args.description_file:
        description = parse_description_file(args.description_file)

    # Parse tags
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    url = upload_video(
        video_path=args.video,
        title=args.title,
        description=description,
        tags=tags,
        category_id=args.category,
        privacy=args.privacy,
        client_secrets=args.client_secrets,
    )

    print(f"\nVideo live at: {url}")


if __name__ == "__main__":
    main()
