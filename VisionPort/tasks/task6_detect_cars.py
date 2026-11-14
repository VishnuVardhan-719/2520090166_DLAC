"""Task 6: Detect cars in a traffic video using YOLO.

Task No: 6
Description: Detect cars in a traffic video using YOLO.
Expected Output: Car detection with bounding boxes
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

# COCO Class ID for car is 2
CAR_CLASS_ID = 2


def main(source_path: Path | None = None, conf_thresh: float = 0.25, headless: bool = False, max_frames: int | None = None):
    print("=" * 65)
    print(" VisionPort – Task 6: Detect Cars in Traffic Video Using YOLO")
    print("=" * 65)

    target_source = source_path or (SAMPLES_DIR / "traffic_sample.mp4")
    if not target_source.exists():
        target_source = SAMPLES_DIR / "traffic_intersection.jpg"

    print(f"Loading YOLOv8 model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # Check if input is image or video
    is_video = target_source.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")

    if not is_video:
        print(f"Running car detection on image: {target_source}")
        img = cv2.imread(str(target_source))
        if img is None:
            print(f"[ERROR] Could not read image: {target_source}")
            return 1

        results = model.predict(img, classes=[CAR_CLASS_ID], conf=conf_thresh, verbose=False)
        annotated = img.copy()
        car_count = 0

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            car_count += 1

            # Draw cyan bounding box for car
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 2)
            label = f"CAR {conf*100:.0f}%"
            cv2.putText(annotated, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2)

        cv2.putText(
            annotated,
            f"VisionPort Task 6: YOLO Car Detection | Cars Detected: {car_count}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if not headless:
            cv2.imshow("VisionPort Task 6 - Car Detection Using YOLO", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return 0

    print(f"Running car detection on video: {target_source}")
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
        results = model.predict(frame, classes=[CAR_CLASS_ID], conf=conf_thresh, verbose=False)
        annotated = frame.copy()
        car_count = 0

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            car_count += 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(annotated, f"CAR {conf*100:.0f}%", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2)

        cv2.putText(
            annotated,
            f"VisionPort Task 6: YOLO Car Detection | Frame: {frame_idx} | Cars: {car_count}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if not headless:
            cv2.imshow("VisionPort Task 6 - Car Detection Using YOLO (Press Q to Quit)", annotated)
            key = cv2.waitKey(frame_delay) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print(f"Processed {frame_idx} video frames with YOLO car detection.")
    print("Expected Output: Car detection with bounding boxes [COMPLETED].")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 6: Detect Cars in Video Using YOLO")
    parser.add_argument("--source", type=Path, default=None, help="Input video or image path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed")
    args = parser.parse_args()

    sys.exit(main(source_path=args.source, conf_thresh=args.conf, headless=args.headless, max_frames=args.max_frames))

