"""Task 4: Load and play a traffic video using OpenCV.

Task No: 4
Description: Load and play a traffic video using OpenCV.
Expected Output: Video playback
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"


def main(video_path: Path | None = None, headless: bool = False, max_frames: int | None = None):
    print("=" * 65)
    print(" VisionPort – Task 4: Load and Play Traffic Video")
    print("=" * 65)

    target_video = video_path or (SAMPLES_DIR / "traffic_sample.mp4")

    print(f"Loading video from: {target_video}")
    cap = cv2.VideoCapture(str(target_video))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {target_video}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print("Video Properties:")
    print(f"  Resolution  : {width} x {height}")
    print(f"  Frame Rate  : {fps:.2f} FPS")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration    : {duration:.2f} seconds")

    if not headless:
        print("Playing video... Press [Q] or [ESC] to stop playback.")

    frame_delay = max(1, int(1000 / fps))
    frame_count = 0
    saved_preview = False

    while cap.isOpened():
        if max_frames and frame_count >= max_frames:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_count += 1
        annotated = frame.copy()
        timestamp = frame_count / fps

        # HUD Overlay
        cv2.putText(
            annotated,
            f"VisionPort Task 4: Video Playback | Frame {frame_count}/{total_frames} ({timestamp:.1f}s) | FPS: {fps:.1f}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 200),
            2,
            cv2.LINE_AA,
        )

        if not headless:
            cv2.imshow("VisionPort Task 4 - Video Playback (Press Q to Quit)", annotated)
            key = cv2.waitKey(frame_delay) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("Playback stopped by user.")
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print(f"Played {frame_count} frames. Expected output: Video playback [COMPLETED].")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 4: Load and Play Traffic Video")
    parser.add_argument("--source", type=Path, default=None, help="Input video path")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit maximum frames played")
    args = parser.parse_args()

    sys.exit(main(video_path=args.source, headless=args.headless, max_frames=args.max_frames))

