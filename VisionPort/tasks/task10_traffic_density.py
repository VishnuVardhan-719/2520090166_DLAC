"""Task 10: Calculate traffic density based on vehicle count.

Task No: 10
Description: Calculate traffic density based on vehicle count.
Expected Output: Low, medium and high traffic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"

VEHICLE_CLASSES = [1, 2, 3, 5, 7]


def classify_density(count: int) -> tuple[str, tuple[int, int, int]]:
    """Classify vehicle count into LOW, MEDIUM, or HIGH density."""
    if count <= 5:
        return "LOW", (0, 220, 0)       # Green
    elif count <= 12:
        return "MEDIUM", (0, 215, 255)  # Amber
    else:
        return "HIGH", (0, 0, 255)      # Red


def main(source_path: Path | None = None, conf_thresh: float = 0.25, headless: bool = False, max_frames: int | None = None):
    print("=" * 65)
    print(" VisionPort – Task 10: Calculate Traffic Density")
    print("=" * 65)

    target_source = source_path or (SAMPLES_DIR / "traffic_sample.mp4")
    if not target_source.exists():
        target_source = SAMPLES_DIR / "traffic_congested.jpg"

    print(f"Loading YOLOv8 model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    is_video = target_source.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")

    if not is_video:
        img = cv2.imread(str(target_source))
        if img is None:
            print(f"[ERROR] Could not read image: {target_source}")
            return 1

        results = model.predict(img, classes=VEHICLE_CLASSES, conf=conf_thresh, verbose=False)
        annotated = img.copy()
        count = len(results[0].boxes)
        density_label, density_color = classify_density(count)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), density_color, 2)

        h, w = img.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, 55), (20, 20, 25), -1)
        cv2.putText(
            annotated,
            f"TASK 10: TRAFFIC DENSITY = {density_label} | Vehicles Count: {count}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            density_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            "Classification: LOW (<=5 vehicles), MEDIUM (6-12 vehicles), HIGH (>12 vehicles)",
            (16, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        print(f"Vehicle Count  : {count}")
        print(f"Traffic Density: {density_label}")

        if not headless:
            cv2.imshow("VisionPort Task 10 - Traffic Density", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return 0

    cap = cv2.VideoCapture(str(target_source))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {target_source}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_delay = max(1, int(1000 / fps))
    frame_idx = 0
    saved_snapshot = False

    while cap.isOpened():
        if max_frames and frame_idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1
        results = model.predict(frame, classes=VEHICLE_CLASSES, conf=conf_thresh, verbose=False)
        annotated = frame.copy()
        count = len(results[0].boxes)
        density_label, density_color = classify_density(count)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), density_color, 2)

        h, w = frame.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, 50), (20, 20, 25), -1)
        cv2.putText(
            annotated,
            f"TASK 10: FRAME {frame_idx} | DENSITY = {density_label} (Vehicles: {count})",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            density_color,
            2,
            cv2.LINE_AA,
        )



        if not headless:
            cv2.imshow("VisionPort Task 10 - Traffic Density (Press Q to Quit)", annotated)
            key = cv2.waitKey(frame_delay) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames. Expected Output: Low, medium and high traffic [COMPLETED].")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 10: Calculate Traffic Density")
    parser.add_argument("--source", type=Path, default=None, help="Input video or image path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed")
    args = parser.parse_args()

    sys.exit(main(source_path=args.source, conf_thresh=args.conf, headless=args.headless, max_frames=args.max_frames))

