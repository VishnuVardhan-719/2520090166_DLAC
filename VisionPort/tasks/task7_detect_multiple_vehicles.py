"""Task 7: Detect bikes, buses and trucks.

Task No: 7
Description: Detect bikes, buses and trucks.
Expected Output: Multiple vehicle detection
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

# COCO: 1=bicycle, 3=motorcycle, 5=bus, 7=truck
MULTI_VEHICLE_CLASSES = [1, 3, 5, 7]

CLASS_COLOR_MAP = {
    "bicycle": (50, 255, 150),     # Light Green
    "motorcycle": (0, 220, 0),      # Bright Green
    "bus": (255, 80, 180),          # Magenta
    "truck": (0, 140, 255),         # Orange
}


def main(source_path: Path | None = None, conf_thresh: float = 0.25, headless: bool = False, max_frames: int | None = None):
    print("=" * 65)
    print(" VisionPort – Task 7: Detect Bikes, Buses and Trucks")
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

        results = model.predict(img, classes=MULTI_VEHICLE_CLASSES, conf=conf_thresh, verbose=False)
        annotated = img.copy()
        counts = {"bike": 0, "bus": 0, "truck": 0}

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name = results[0].names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cat = "bike" if name in ("bicycle", "motorcycle") else name
            counts[cat] = counts.get(cat, 0) + 1
            color = CLASS_COLOR_MAP.get(name, (0, 255, 255))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{name.upper()} {conf*100:.0f}%"
            cv2.putText(annotated, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(
            annotated,
            f"VisionPort Task 7: Multi-Vehicle | Bikes: {counts['bike']} | Buses: {counts['bus']} | Trucks: {counts['truck']}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        print(f"Detections -> Bikes: {counts['bike']}, Buses: {counts['bus']}, Trucks: {counts['truck']}")

        if not headless:
            cv2.imshow("VisionPort Task 7 - Multiple Vehicle Detection", annotated)
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
        results = model.predict(frame, classes=MULTI_VEHICLE_CLASSES, conf=conf_thresh, verbose=False)
        annotated = frame.copy()
        counts = {"bike": 0, "bus": 0, "truck": 0}

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name = results[0].names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cat = "bike" if name in ("bicycle", "motorcycle") else name
            counts[cat] = counts.get(cat, 0) + 1
            color = CLASS_COLOR_MAP.get(name, (0, 255, 255))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{name.upper()} {conf*100:.0f}%", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(
            annotated,
            f"VisionPort Task 7: Multi-Vehicle | Bikes: {counts['bike']} | Buses: {counts['bus']} | Trucks: {counts['truck']}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )



        if not headless:
            cv2.imshow("VisionPort Task 7 - Multiple Vehicle Detection (Press Q to Quit)", annotated)
            key = cv2.waitKey(frame_delay) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print(f"Processed {frame_idx} video frames. Expected Output: Multiple vehicle detection [COMPLETED].")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 7: Detect Bikes, Buses and Trucks")
    parser.add_argument("--source", type=Path, default=None, help="Input video or image path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed")
    args = parser.parse_args()

    sys.exit(main(source_path=args.source, conf_thresh=args.conf, headless=args.headless, max_frames=args.max_frames))

