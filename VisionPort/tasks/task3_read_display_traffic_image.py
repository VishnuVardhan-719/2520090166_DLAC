"""Task 3: Read and display traffic images using OpenCV.

Task No: 3
Description: Read and display traffic images using OpenCV.
Expected Output: Image display window
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"


def main(image_path: Path | None = None, headless: bool = False):
    print("=" * 65)
    print(" VisionPort – Task 3: Read and Display Traffic Images")
    print("=" * 65)

    target_path = image_path or (SAMPLES_DIR / "traffic_intersection.jpg")

    print(f"Reading image from: {target_path}")
    img = cv2.imread(str(target_path))
    if img is None:
        print(f"[ERROR] Could not read image from: {target_path}")
        return 1

    h, w, c = img.shape
    file_size_kb = round(target_path.stat().st_size / 1024, 1)
    aspect_ratio = round(w / max(1, h), 2)

    print("Image Metadata:")
    print(f"  Resolution  : {w} x {h} pixels")
    print(f"  Channels    : {c} (BGR Format)")
    print(f"  Data Type   : {img.dtype}")
    print(f"  Aspect Ratio: {aspect_ratio} : 1")
    print(f"  File Size   : {file_size_kb} KB")
    print(f"  Pixel Range : [{img.min()}, {img.max()}]")

    annotated = img.copy()
    cv2.putText(
        annotated,
        f"VisionPort Task 3: {target_path.name} | Res: {w}x{h} | Channels: {c}",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if not headless:
        print("Opening OpenCV Image Display Window. Press any key to close...")
        try:
            cv2.imshow("VisionPort Task 3 - Image Display Window", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    print("Task 3 completed successfully. Expected Output: Image display window.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3: Read and Display Traffic Images")
    parser.add_argument("--source", type=Path, default=None, help="Input image path")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    args = parser.parse_args()

    sys.exit(main(image_path=args.source, headless=args.headless))

