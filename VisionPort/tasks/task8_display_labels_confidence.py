"""Task 8: Display vehicle names and confidence scores.

Task No: 8
Description: Display vehicle names and confidence scores.
Expected Output: Labelled detection output
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

VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

COLOR_MAP = {
    "car": (255, 180, 0),        # Cyan
    "bicycle": (50, 255, 150),   # Light Green
    "motorcycle": (0, 220, 0),    # Bright Green
    "bus": (255, 80, 180),        # Magenta
    "truck": (0, 140, 255),       # Orange
}


def main(source_path: Path | None = None, conf_thresh: float = 0.25, headless: bool = False, max_frames: int | None = None):
    print("=" * 65)
    print(" VisionPort – Task 8: Display Vehicle Names & Confidence Scores")
    print("=" * 65)

    target_source = source_path or (SAMPLES_DIR / "traffic_sample.mp4")
    if not target_source.exists():
        target_source = SAMPLES_DIR / "traffic_intersection.jpg"

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
        print("\nDetected Vehicles with Confidence Scores:")
        print(f"  {'#':<4} {'Class Name':<16} {'Confidence Score':<18} {'Bounding Box [x1,y1,x2,y2]'}")
        print("  " + "-" * 60)

        det_count = 0
        for box in results[0].boxes:
            det_count += 1
            cls_id = int(box.cls[0])
            name = results[0].names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLOR_MAP.get(name, (0, 255, 255))

            # Print to console
            print(f"  {det_count:<4} {name:<16} {conf*100:.2f}% ({conf:.3f})     [{x1}, {y1}, {x2}, {y2}]")

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # High-contrast label plaque
            label = f"{name.upper()} | {conf*100:.1f}%"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(th + 8, y1)
            cv2.rectangle(annotated, (x1, ty - th - 6), (x1 + tw + 6, ty + 2), color, -1)
            cv2.putText(annotated, label, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(
            annotated,
            f"VisionPort Task 8: Labelled Output | Total Vehicles: {det_count}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        print("-" * 65)
        print(f"Total vehicles labelled: {det_count}")

        if not headless:
            cv2.imshow("VisionPort Task 8 - Labelled Detections", annotated)
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
        det_count = 0

        for box in results[0].boxes:
            det_count += 1
            cls_id = int(box.cls[0])
            name = results[0].names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLOR_MAP.get(name, (0, 255, 255))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{name.upper()} | {conf*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(th + 6, y1)
            cv2.rectangle(annotated, (x1, ty - th - 6), (x1 + tw + 6, ty + 2), color, -1)
            cv2.putText(annotated, label, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        cv2.putText(
            annotated,
            f"VisionPort Task 8: Labelled Output | Frame: {frame_idx} | Detections: {det_count}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )



        if not headless:
            cv2.imshow("VisionPort Task 8 - Labelled Detections (Press Q to Quit)", annotated)
            key = cv2.waitKey(frame_delay) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print(f"Processed {frame_idx} video frames. Expected Output: Labelled detection output [COMPLETED].")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 8: Display Vehicle Names and Confidence Scores")
    parser.add_argument("--source", type=Path, default=None, help="Input video or image path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed")
    args = parser.parse_args()

    sys.exit(main(source_path=args.source, conf_thresh=args.conf, headless=args.headless, max_frames=args.max_frames))

