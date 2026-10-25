"""Upload a local file to Runway as an ephemeral asset using the runwayml SDK.

See https://docs.dev.runwayml.com/assets/uploads/

Requires the RUNWAYML_API_SECRET environment variable (loaded from a local
.env file if present).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from runwayml import RunwayML

DEFAULT_VIDEO_PATH = Path("video_input/Compositing_Upload_V2.mp4")

# Container format -> expected Content-Type, per https://docs.dev.runwayml.com/assets/inputs
VIDEO_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".3gp": "video/3gpp",
    ".ogv": "video/ogg",
}


def resolve_video_content_type(file_path: Path) -> str:
    """Look up the Content-Type Runway expects for `file_path`'s extension."""
    content_type = VIDEO_CONTENT_TYPES.get(file_path.suffix.lower())
    if content_type is None:
        supported = ", ".join(sorted(VIDEO_CONTENT_TYPES))
        raise ValueError(
            f"Unsupported video extension '{file_path.suffix}' for {file_path.name}. "
            f"Supported extensions: {supported}"
        )
    return content_type


def upload_ephemeral_asset(file_path: Path) -> str:
    """Upload `file_path` as an ephemeral asset and return its runway:// URI."""
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    content_type = resolve_video_content_type(file_path)

    client = RunwayML()
    with file_path.open("rb") as f:
        response = client.uploads.create_ephemeral(file=(file_path.name, f, content_type))
    return response.uri


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Upload a file to Runway as an ephemeral asset.")
    parser.add_argument(
        "file",
        nargs="?",
        default=str(DEFAULT_VIDEO_PATH),
        help=f"Path to the file to upload (default: {DEFAULT_VIDEO_PATH})",
    )
    args = parser.parse_args()

    try:
        runway_uri = upload_ephemeral_asset(Path(args.file))
    except Exception as exc:  # FileNotFoundError or runwayml.APIError subclasses
        sys.exit(f"Upload failed: {exc}")

    print("Upload succeeded.")
    print(f"runway URI (valid 24h): {runway_uri}")


if __name__ == "__main__":
    main()
