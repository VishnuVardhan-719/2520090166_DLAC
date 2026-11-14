"""Task 13: Calculate approximate vehicle speed.

Task No: 13
Description: Calculate approximate vehicle speed.
Expected Output: Speed estimation
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"

VEHICLE_CLASSES = [1, 2, 3, 5, 7]


def main(source_path: Path | None = None, pixels_per_meter: float = 12.0, conf_thresh: float = 0.25, headless: bool = False, max_frames: int | None = None):
    print("=" * 65)
    print(" VisionPort – Task 13: Calculate Approximate Vehicle Speed")
    print("=" * 65)

    target_source = source_path or (SAMPLES_DIR / "traffic_sample.mp4")
    if not target_source.exists():
        target_source = SAMPLES_DIR / "traffic_sample.mp4"

    print(f"Loading YOLOv8 model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(target_source))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {target_source}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_delay = max(1, int(1000 / fps))
    frame_idx = 0
    saved_snapshot = False

    # Track centroid history: {track_id: deque([(frame_idx, cx, cy)])}
    track_history = defaultdict(lambda: deque(maxlen=8))
    smoothed_speeds = {}

    print(f"Video FPS: {fps:.1f} | Calibrated Scale: {pixels_per_meter} pixels/meter")

    while cap.isOpened():
        if max_frames and frame_idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1
        results = model.track(frame, persist=True, classes=VEHICLE_CLASSES, conf=conf_thresh, verbose=False)
        annotated = frame.copy()

        current_speeds = []

        if results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            clses = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, tid, cls_id in zip(boxes, track_ids, clses):
                x1, y1, x2, y2 = [int(round(float(v))) for v in box]
                tid = int(tid)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                hist = track_history[tid]
                hist.append((frame_idx, cx, cy))

                speed_kmh = 0.0
                if len(hist) >= 2:
                    f0, x0, y0 = hist[0]
                    f1, x1, y1_pt = hist[-1]
                    dt_sec = max(1, f1 - f0) / fps
                    dist_px = float(np.hypot(x1 - x0, y1_pt - y0))
                    dist_m = dist_px / pixels_per_meter
                    speed_kmh = (dist_m / dt_sec) * 3.6

                    # Smoothing
                    if tid in smoothed_speeds:
                        speed_kmh = 0.6 * speed_kmh + 0.4 * smoothed_speeds[tid]
                    smoothed_speeds[tid] = speed_kmh
                else:
                    speed_kmh = smoothed_speeds.get(tid, 0.0)

                current_speeds.append(speed_kmh)
                name = str(results[0].names[int(cls_id)])

                # Draw vehicle bounding box and speed plaque
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 180), 2)
                label = f"ID:{tid} {name.upper()} | {speed_kmh:.1f} km/h"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (int(x1), int(y1) - th - 6), (int(x1) + tw + 6, int(y1)), (0, 255, 180), -1)
                cv2.putText(annotated, label, (int(x1) + 3, int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        avg_speed = sum(current_speeds) / len(current_speeds) if current_speeds else 0.0

        # Top Banner
        h, w = frame.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, 50), (20, 20, 25), -1)
        cv2.putText(
            annotated,
            f"TASK 13: SPEED ESTIMATION | Avg Speed: {avg_speed:.1f} km/h | Active Vehicles: {len(current_speeds)}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 180),
            2,
            cv2.LINE_AA,
        )



        if not headless:
            cv2.imshow("VisionPort Task 13 - Speed Estimation (Press Q to Quit)", annotated)
            key = cv2.waitKey(frame_delay) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print(f"Speed estimation complete across {frame_idx} frames. Final avg speed: {avg_speed:.1f} km/h.")
    print("Expected Output: Speed estimation [COMPLETED].")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 13: Calculate Approximate Vehicle Speed")
    parser.add_argument("--source", type=Path, default=None, help="Input video path")
    parser.add_argument("--ppm", type=float, default=12.0, help="Pixels per meter calibration")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed")
    args = parser.parse_args()

    sys.exit(main(source_path=args.source, pixels_per_meter=args.ppm, conf_thresh=args.conf, headless=args.headless, max_frames=args.max_frames))

