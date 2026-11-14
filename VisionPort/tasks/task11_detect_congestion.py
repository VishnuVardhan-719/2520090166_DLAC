"""Task 11: Detect traffic congestion at signal junctions.

Task No: 11
Description: Detect traffic congestion at signal junctions.
Expected Output: Congestion alerts
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
CONGESTION_THRESHOLD = 8


def main(source_path: Path | None = None, conf_thresh: float = 0.25, headless: bool = False, max_frames: int | None = None):
    print("=" * 65)
    print(" VisionPort – Task 11: Detect Junction Traffic Congestion Alerts")
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
        is_congested = count >= CONGESTION_THRESHOLD
        h, w = img.shape[:2]

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255) if is_congested else (0, 255, 0), 2)

        if is_congested:
            # Red Congestion Warning Banner
            cv2.rectangle(annotated, (0, 0), (w, 55), (0, 0, 200), -1)
            cv2.putText(
                annotated,
                f"WARNING: JUNCTION CONGESTION ALERT ACTIVE! ({count} Vehicles >= {CONGESTION_THRESHOLD})",
                (16, 35),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(annotated, (4, 4), (w - 5, h - 5), (0, 0, 255), 4)
            print(f"⚠️ [CONGESTION ALERT TRIGGERED] Vehicles: {count} >= Threshold {CONGESTION_THRESHOLD}")
        else:
            cv2.rectangle(annotated, (0, 0), (w, 45), (20, 20, 25), -1)
            cv2.putText(
                annotated,
                f"TASK 11: JUNCTION CLEAR | Vehicles Count = {count} (Normal Flow)",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            print(f"[STATUS] Junction clear. Vehicles: {count} < Threshold {CONGESTION_THRESHOLD}")



        if not headless:
            cv2.imshow("VisionPort Task 11 - Congestion Alerts", annotated)
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
        is_congested = count >= CONGESTION_THRESHOLD
        h, w = frame.shape[:2]

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255) if is_congested else (0, 255, 0), 2)

        if is_congested:
            cv2.rectangle(annotated, (0, 0), (w, 50), (0, 0, 200), -1)
            cv2.putText(
                annotated,
                f"WARNING: JUNCTION CONGESTION ALERT! (Vehicles: {count} >= {CONGESTION_THRESHOLD})",
                (16, 32),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(annotated, (4, 4), (w - 5, h - 5), (0, 0, 255), 4)
        else:
            cv2.rectangle(annotated, (0, 0), (w, 40), (20, 20, 25), -1)
            cv2.putText(
                annotated,
                f"TASK 11: JUNCTION CLEAR | Frame {frame_idx} (Vehicles: {count})",
                (16, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )



        if not headless:
            cv2.imshow("VisionPort Task 11 - Congestion Alerts (Press Q to Quit)", annotated)
            key = cv2.waitKey(frame_delay) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames. Expected Output: Congestion alerts [COMPLETED].")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 11: Detect Traffic Congestion Alerts")
    parser.add_argument("--source", type=Path, default=None, help="Input video or image path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed")
    args = parser.parse_args()

    sys.exit(main(source_path=args.source, conf_thresh=args.conf, headless=args.headless, max_frames=args.max_frames))

