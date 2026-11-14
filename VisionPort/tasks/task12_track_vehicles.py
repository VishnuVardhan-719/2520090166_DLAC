"""Task 12: Track individual vehicles across video frames.

Task No: 12
Description: Track individual vehicles across video frames.
Expected Output: Vehicle tracking IDs
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


def main(source_path: Path | None = None, conf_thresh: float = 0.25, headless: bool = False, max_frames: int | None = None):
    print("=" * 65)
    print(" VisionPort – Task 12: Track Individual Vehicles Across Frames")
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

    # Centroid history for motion trails (maxlen=25 points)
    track_trails = defaultdict(lambda: deque(maxlen=25))
    unique_ids_seen = set()

    print("Running ByteTrack Multi-Object Tracking...")

    while cap.isOpened():
        if max_frames and frame_idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_idx += 1
        # Ultralytics ByteTrack tracking
        results = model.track(frame, persist=True, classes=VEHICLE_CLASSES, conf=conf_thresh, verbose=False)
        annotated = frame.copy()

        active_track_ids = []
        if results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            clses = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, tid, cls_id in zip(boxes, track_ids, clses):
                active_track_ids.append(tid)
                unique_ids_seen.add(tid)
                x1, y1, x2, y2 = [int(round(float(v))) for v in box]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                track_trails[tid].append((cx, cy))

                name = results[0].names[cls_id]

                # Draw bounding box & ID plaque
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                label = f"ID:{tid} {name.upper()}"
                cv2.rectangle(annotated, (x1, y1 - 22), (x1 + len(label) * 9, y1), (0, 255, 255), -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

                # Draw motion trail
                pts = np.array(track_trails[tid], dtype=np.int32)
                if len(pts) >= 2:
                    cv2.polylines(annotated, [pts], isClosed=False, color=(0, 220, 255), thickness=2)

        # Top HUD Banner
        h, w = frame.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, 50), (20, 20, 25), -1)
        cv2.putText(
            annotated,
            f"TASK 12: TRACKING | Frame: {frame_idx} | Active Vehicles: {len(active_track_ids)} | Unique IDs Seen: {len(unique_ids_seen)}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )



        if not headless:
            cv2.imshow("VisionPort Task 12 - Vehicle Tracking (Press Q to Quit)", annotated)
            key = cv2.waitKey(frame_delay) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print(f"Tracking complete: {len(unique_ids_seen)} unique vehicle IDs tracked across {frame_idx} frames.")
    print("Expected Output: Vehicle tracking IDs [COMPLETED].")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 12: Track Individual Vehicles Across Frames")
    parser.add_argument("--source", type=Path, default=None, help="Input video path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed")
    args = parser.parse_args()

    sys.exit(main(source_path=args.source, conf_thresh=args.conf, headless=args.headless, max_frames=args.max_frames))

