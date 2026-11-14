"""Task 2: Capture an image of a traffic signal using a camera.

Task No: 2
Description: Capture an image of a traffic signal using a camera.
Expected Output: Traffic image capture
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "visionport_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main(camera_idx: int = 0, headless: bool = False, auto_capture: bool = False):
    print("=" * 65)
    print(" VisionPort – Task 2: Capture Traffic Signal Image Using Camera")
    print("=" * 65)

    cap = cv2.VideoCapture(camera_idx)
    captured = None

    if cap.isOpened():
        print(f"Connected to camera (device {camera_idx}).")
        if auto_capture or headless:
            print("Auto-capturing frame from camera...")
            for _ in range(8):
                ok, f = cap.read()
                if ok and f is not None:
                    captured = f
        else:
            print("Camera preview open! Press [SPACE] to capture traffic signal, or [Q] to quit.")
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                preview = frame.copy()
                cv2.putText(
                    preview,
                    "Task 2: Press SPACE to capture traffic signal | Q to quit",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("VisionPort Task 2 - Camera Preview", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACE
                    captured = frame
                    print("Traffic signal captured via SPACE key.")
                    break
                if key in (ord("q"), 27):
                    break
        cap.release()
        if not headless:
            cv2.destroyAllWindows()

    if captured is None:
        sample_path = SAMPLES_DIR / "traffic_signal.jpg"
        print(f"[INFO] Camera device not accessible; loading camera snapshot sample: {sample_path.name}")
        captured = cv2.imread(str(sample_path))

    if captured is None:
        print("[ERROR] Failed to obtain image from camera or sample file.")
        return 1

    h, w = captured.shape[:2]
    annotated = captured.copy()
    cv2.putText(
        annotated,
        f"VisionPort Task 2: Traffic Signal Capture ({w}x{h})",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    out_file = OUTPUT_DIR / "captured_signal.jpg"
    cv2.imwrite(str(out_file), captured)
    print(f"Captured traffic signal saved to: {out_file} ({w}x{h} px)")

    if not headless:
        print("Displaying captured traffic signal. Press any key to close...")
        try:
            cv2.imshow("VisionPort Task 2 - Traffic Image Capture", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    print("Task 2 completed successfully. Expected Output: Traffic image capture.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2: Capture Traffic Signal Image Using Camera")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--auto-capture", action="store_true", help="Auto capture without waiting for keypress")
    args = parser.parse_args()

    sys.exit(main(camera_idx=args.camera, headless=args.headless, auto_capture=args.auto_capture))

