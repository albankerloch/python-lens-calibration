"""Run a video-to-video generation with Runway's Aleph 2.0 model.

Uploads a local video as an ephemeral asset (see runway_ephemeral_upload.py),
sends it to the `aleph2` model via client.video_to_video.create(), waits for
the task to complete, then downloads the resulting file(s).

Requires the RUNWAYML_API_SECRET environment variable (loaded from a local
.env file if present).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from runwayml import RunwayML, TaskFailedError, TaskTimeoutError

from runway_ephemeral_upload import upload_ephemeral_asset

DEFAULT_VIDEO_PATH = Path("video_input/Compositing_Upload_V2.mp4")
OUTPUT_DIR = Path("video_output")
DEFAULT_PROMPT = "Enhance the lighting and add a cinematic look"
DEFAULT_SEED = 42
DEFAULT_TIMEOUT = 1800  # seconds; 0 means wait indefinitely

OUTPUT_FORMATS = ["mp4", "prores", "png_sequence"]
PRORES_PROFILES = ["422", "4444", "422 Proxy", "422 LT", "422 HQ", "4444 XQ"]
# outputFormat -> default extension for the main output file
OUTPUT_FORMAT_EXTENSIONS = {"mp4": ".mp4", "prores": ".mov", "png_sequence": ".zip"}
LAST_TASK_ID_FILE = OUTPUT_DIR / ".last_task_id"


def save_last_task_id(task_id: str) -> None:
    LAST_TASK_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_TASK_ID_FILE.write_text(task_id, encoding="utf-8")


def run_aleph_video_to_video(
    client: RunwayML,
    video_path: Path,
    prompt_text: str,
    ratio: str | None,
    seed: int | None,
    output_format: str,
    prores_profile: str | None,
    timeout: float | None,
) -> list[str]:
    """Upload `video_path`, run it through aleph2, and return the output URLs."""
    print(f"Uploading {video_path} as an ephemeral asset...")
    video_uri = upload_ephemeral_asset(video_path)
    print(f"Uploaded. runway URI: {video_uri}")

    print(f"Starting aleph2 video_to_video task (output format: {output_format})...")
    create_kwargs = {
        "model": "aleph2",
        "video_uri": video_uri,
        "prompt_text": prompt_text,
        "output_format": output_format,
    }
    if ratio:
        create_kwargs["ratio"] = ratio
    if seed is not None:
        create_kwargs["seed"] = seed
    if prores_profile:
        create_kwargs["prores_profile"] = prores_profile
    task = client.video_to_video.create(**create_kwargs)
    print(f"Task created: {task.id}. Waiting for completion...")
    save_last_task_id(task.id)

    task_details = task.wait_for_task_output(timeout=timeout)
    print(f"Task succeeded. Credits consumed: {task_details.cost.credits}")
    return task_details.output


def download_output(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Streamed: ProRes outputs can weigh several GB.
    with requests.get(url, timeout=120, stream=True) as response:
        response.raise_for_status()
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)


def download_outputs(output_urls: list[str], base_output_path: Path, output_format: str) -> list[Path]:
    """Download every output URL. `png_sequence` tasks can return a .zip plus a .wav."""
    fallback_ext = OUTPUT_FORMAT_EXTENSIONS[output_format]
    saved_paths = []
    for i, url in enumerate(output_urls):
        if i == 0:
            path = base_output_path
        else:
            url_ext = Path(urlparse(url).path).suffix or fallback_ext
            path = base_output_path.with_name(f"{base_output_path.stem}_{i}{url_ext}")
        print(f"Downloading output to {path}...")
        download_output(url, path)
        saved_paths.append(path)
    return saved_paths


def check_task(client: RunwayML, task_id: str, output: str | None) -> None:
    """Look up an existing task by ID and download its output if it succeeded."""
    task_details = client.tasks.retrieve(task_id)
    print(f"Task {task_id}: status={task_details.status}")

    if task_details.status == "SUCCEEDED":
        print(f"Credits consumed: {task_details.cost.credits}")
        if not task_details.output:
            print("Task succeeded but returned no output.")
            return
        if output:
            base_output_path = Path(output)
        else:
            first_ext = Path(urlparse(task_details.output[0]).path).suffix or ".mp4"
            base_output_path = OUTPUT_DIR / f"{task_id}_output{first_ext}"
        saved_paths = download_outputs(task_details.output, base_output_path, "mp4")
        print("Done. Saved: " + ", ".join(str(p) for p in saved_paths))
    elif task_details.status == "FAILED":
        print(f"Credits consumed: {task_details.cost.credits}")
        print(f"Failure: {task_details.failure}")
    elif task_details.status == "CANCELLED":
        print("Task was cancelled.")
    else:
        # PENDING, RUNNING, or THROTTLED
        progress = getattr(task_details, "progress", None)
        if progress is not None:
            print(f"Progress: {progress:.0%}")
        print(f"Estimated cost so far: {task_details.estimated_cost.credits} credits")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run Runway's Aleph 2.0 video-to-video model on a local video.")
    parser.add_argument(
        "--check-task",
        default=None,
        metavar="TASK_ID",
        help="Look up an existing task by ID instead of starting a new generation "
        "(use 'last' for the most recently created task), then exit.",
    )
    parser.add_argument(
        "--video",
        default=str(DEFAULT_VIDEO_PATH),
        help=f"Path to the input video (default: {DEFAULT_VIDEO_PATH})",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        default=None,
        help=f"Text prompt describing the desired edit (default: {DEFAULT_PROMPT!r})",
    )
    prompt_group.add_argument(
        "--prompt-file",
        default=None,
        help="Path to a text file containing the prompt, e.g. prompt_input/realistic.txt",
    )
    parser.add_argument("--ratio", default=None, help="Output ratio, e.g. 1280:720 (default: unset)")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for reproducible results (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--output-format",
        choices=OUTPUT_FORMATS,
        default="mp4",
        help="Output container: mp4 (default), prores, or png_sequence. "
        "Non-mp4 formats add a 5 credits/s surcharge.",
    )
    parser.add_argument(
        "--prores-profile",
        choices=PRORES_PROFILES,
        default=None,
        help="ProRes profile, only used when --output-format prores (API default: '4444')",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Path to save the generated file (default: {OUTPUT_DIR}/<input name>_output<ext>)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Max seconds to wait for the task, 0 = wait indefinitely (default: {DEFAULT_TIMEOUT})",
    )
    args = parser.parse_args()

    if args.check_task:
        client = RunwayML()
        if args.check_task == "last":
            if not LAST_TASK_ID_FILE.is_file():
                sys.exit(f"No last task ID found at {LAST_TASK_ID_FILE}. Pass an explicit task ID instead.")
            task_id = LAST_TASK_ID_FILE.read_text(encoding="utf-8").strip()
        else:
            task_id = args.check_task
        check_task(client, task_id, args.output)
        return

    if args.prores_profile and args.output_format != "prores":
        parser.error("--prores-profile requires --output-format prores")

    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    elif args.prompt:
        prompt_text = args.prompt
    else:
        prompt_text = DEFAULT_PROMPT

    client = RunwayML()

    try:
        output_urls = run_aleph_video_to_video(
            client,
            Path(args.video),
            prompt_text,
            args.ratio,
            args.seed,
            args.output_format,
            args.prores_profile,
            args.timeout or None,
        )
    except TaskFailedError as exc:
        sys.exit(f"Task failed (credits consumed: {exc.task_details.cost.credits}): {exc.task_details.failure}")
    except TaskTimeoutError as exc:
        details = exc.task_details
        progress = getattr(details, "progress", None)
        progress_str = f", progress: {progress:.0%}" if progress is not None else ""
        sys.exit(
            f"Timed out waiting for task {details.id} (status: {details.status}{progress_str}).\n"
            "The task keeps running on Runway's side and will still consume credits if it completes.\n"
            "Re-run with a longer --timeout, or check its status later with:\n"
            f"  client.tasks.retrieve('{details.id}')"
        )
    except FileNotFoundError as exc:
        sys.exit(f"Upload failed: {exc}")

    if not output_urls:
        sys.exit("Task succeeded but returned no output.")

    if args.output:
        base_output_path = Path(args.output)
    else:
        ext = OUTPUT_FORMAT_EXTENSIONS[args.output_format]
        base_output_path = OUTPUT_DIR / f"{Path(args.video).stem}_output{ext}"

    saved_paths = download_outputs(output_urls, base_output_path, args.output_format)
    print("Done. Saved: " + ", ".join(str(p) for p in saved_paths))


if __name__ == "__main__":
    main()
