"""Task 15: Estimate traffic waiting time.

Task No: 15
Description: Estimate traffic waiting time.
Expected Output: Waiting-time report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"

VEHICLE_CLASSES = [1, 2, 3, 5, 7]


def compute_waiting_time(
    vehicle_count: int,
    signal_status: str = "RED",
    red_remaining_sec: float = 30.0,
    green_duration_sec: float = 30.0,
    red_duration_sec: float = 45.0,
    discharge_rate_vps: float = 0.5,
    lanes: int = 2,
) -> dict:
    """Compute queue clearance time, signal cycles needed, and driver waiting time."""
    capacity_vps = max(0.1, discharge_rate_vps * max(1, lanes))
    queue_clearance_sec = vehicle_count / capacity_vps
    vehicles_per_green = capacity_vps * green_duration_sec
    cycles_needed = max(1, int(np.ceil(vehicle_count / max(1.0, vehicles_per_green)))) if vehicle_count > 0 else 0

    if vehicle_count == 0:
        est_wait_sec = 0.0
    elif signal_status == "RED":
        overflow_cycles = max(0, cycles_needed - 1)
        est_wait_sec = red_remaining_sec + queue_clearance_sec + (overflow_cycles * red_duration_sec)
    elif signal_status == "YELLOW":
        overflow_cycles = max(0, cycles_needed - 1)
        est_wait_sec = red_duration_sec + queue_clearance_sec + (overflow_cycles * red_duration_sec)
    elif signal_status == "GREEN":
        est_wait_sec = queue_clearance_sec * 0.35
    else:
        est_wait_sec = queue_clearance_sec + (red_duration_sec * 0.5)

    density = "LOW" if vehicle_count <= 5 else ("MEDIUM" if vehicle_count <= 12 else "HIGH")

    return {
        "vehicle_count": vehicle_count,
        "signal_status": signal_status,
        "traffic_density": density,
        "queue_clearance_sec": round(queue_clearance_sec, 1),
        "cycles_needed": cycles_needed,
        "estimated_wait_time_sec": round(est_wait_sec, 1),
    }


def main(source_path: Path | None = None, conf_thresh: float = 0.25, headless: bool = False, max_frames: int | None = None):
    print("=" * 65)
    print(" VisionPort – Task 15: Estimate Traffic Waiting Time")
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
        count = len(results[0].boxes)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 180), 2)

        report = compute_waiting_time(vehicle_count=count, signal_status="RED")

        # Top Banner
        h, w = img.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, 60), (20, 20, 25), -1)
        cv2.putText(
            annotated,
            f"TASK 15: WAITING TIME = {report['estimated_wait_time_sec']}s | Clearance Time: {report['queue_clearance_sec']}s",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 200),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Queue: {count} vehicles | Density: {report['traffic_density']} | Signal: {report['signal_status']} | Cycles: {report['cycles_needed']}",
            (16, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        print("Waiting-Time Report:")
        print(json.dumps(report, indent=2))

        if not headless:
            cv2.imshow("VisionPort Task 15 - Waiting Time Estimation", annotated)
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
    all_reports = []

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

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 180), 2)

        # Approximate signal phase from frame index
        phase = "RED" if frame_idx < 42 else ("GREEN" if frame_idx < 90 else "YELLOW")
        report = compute_waiting_time(vehicle_count=count, signal_status=phase)
        all_reports.append(report["estimated_wait_time_sec"])

        h, w = frame.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, 55), (20, 20, 25), -1)
        cv2.putText(
            annotated,
            f"TASK 15: FRAME {frame_idx} | EST. WAIT TIME = {report['estimated_wait_time_sec']}s",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 200),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Vehicles: {count} | Signal: {phase} | Clearance: {report['queue_clearance_sec']}s",
            (16, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )



        if not headless:
            cv2.imshow("VisionPort Task 15 - Waiting Time (Press Q to Quit)", annotated)
            key = cv2.waitKey(frame_delay) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    avg_wait = sum(all_reports) / max(1, len(all_reports))
    final_summary = {
        "frames_evaluated": frame_idx,
        "average_wait_time_sec": round(avg_wait, 1),
        "peak_wait_time_sec": max(all_reports, default=0.0),
    }

    print(f"Processed {frame_idx} video frames. Avg wait time: {avg_wait:.1f}s.")
    print("Waiting-Time Summary:", json.dumps(final_summary))
    print("Expected Output: Waiting-time report [COMPLETED].")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 15: Estimate Traffic Waiting Time")
    parser.add_argument("--source", type=Path, default=None, help="Input video or image path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed")
    args = parser.parse_args()

    sys.exit(main(source_path=args.source, conf_thresh=args.conf, headless=args.headless, max_frames=args.max_frames))

