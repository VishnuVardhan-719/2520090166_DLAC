"""Task 5: Extract individual frames from traffic videos.

Task No: 5
Description: Extract individual frames from traffic videos.
Expected Output: Frames saved as images
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
FRAMES_DIR = Path(tempfile.gettempdir()) / "visionport_extracted_frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def main(video_path: Path | None = None, interval: int = 15, max_saved: int | None = None, headless: bool = False):
    print("=" * 65)
    print(" VisionPort – Task 5: Extract Video Frames")
    print("=" * 65)

    target_video = video_path or (SAMPLES_DIR / "traffic_sample.mp4")

    print(f"Opening video: {target_video}")
    cap = cv2.VideoCapture(str(target_video))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {target_video}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video Info: {width}x{height} @ {fps:.1f} FPS, total {total_frames} frames.")
    print(f"Extracting frame every {interval} frames...")

    saved_frames = []
    frame_idx = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame_idx % max(1, interval) == 0:
            frame_num = len(saved_frames) + 1
            filename = f"frame_{frame_num:04d}.jpg"
            out_path = FRAMES_DIR / filename

            # Annotate saved frame lightly with frame metadata
            annotated = frame.copy()
            cv2.putText(
                annotated,
                f"Task 5 Extracted Frame #{frame_num} (Source Index: {frame_idx})",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(out_path), annotated)
            saved_frames.append(out_path)
            print(f"  [SAVED] {out_path.name} (Time: {frame_idx / fps:.2f}s)")

            if not headless and len(saved_frames) <= 3:
                cv2.imshow("VisionPort Task 5 - Extracted Frame Preview", annotated)
                cv2.waitKey(200)

            if max_saved and len(saved_frames) >= max_saved:
                break

        frame_idx += 1

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print("-" * 65)
    print(f"Extracted and saved {len(saved_frames)} frames into: {FRAMES_DIR}")
    print("Expected Output: Frames saved as images [COMPLETED].")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 5: Extract Individual Frames from Traffic Videos")
    parser.add_argument("--source", type=Path, default=None, help="Input video path")
    parser.add_argument("--interval", type=int, default=15, help="Frame extraction interval")
    parser.add_argument("--max-saved", type=int, default=10, help="Maximum number of frames to save")
    parser.add_argument("--headless", action="store_true", help="Run without GUI preview")
    args = parser.parse_args()

    sys.exit(main(video_path=args.source, interval=args.interval, max_saved=args.max_saved, headless=args.headless))

