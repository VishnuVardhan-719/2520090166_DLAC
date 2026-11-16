"""Task 20: Detect two-wheelers carrying more than two persons (triple-riding).

Task No: 20
Description: Triple-riding detection on two-wheelers.
Expected Output: Triple-riding violation alerts
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"

TWO_WHEELER_CLASSES = [1, 3]  # COCO: bicycle, motorcycle
PERSON_CLASS = 0              # COCO: person
CLASS_NAMES = {1: "bicycle", 3: "motorcycle"}


class CentroidTracker:
    """Simple centroid-based multi-object tracker."""

    def __init__(self, max_dist: float = 75, max_missing: int = 12):
        self.next_id = 1
        self.objects: dict[int, tuple[float, float]] = {}
        self.missing: dict[int, int] = {}
        self.max_dist = max_dist
        self.max_missing = max_missing

    def update(self, boxes_xyxy: list[list[float]]) -> list[int]:
        if not boxes_xyxy:
            for oid in list(self.missing.keys()):
                self.missing[oid] += 1
                if self.missing[oid] > self.max_missing:
                    del self.missing[oid]
                    self.objects.pop(oid, None)
            return []

        centroids = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes_xyxy]

        if not self.objects:
            assigned = []
            for c in centroids:
                oid = self.next_id
                self.next_id += 1
                self.objects[oid] = c
                self.missing[oid] = 0
                assigned.append(oid)
            return assigned

        obj_ids = list(self.objects.keys())
        obj_pts = [self.objects[oid] for oid in obj_ids]

        pairs = []
        for bi, (cx, cy) in enumerate(centroids):
            for oi, (ox, oy) in enumerate(obj_pts):
                d = math.hypot(cx - ox, cy - oy)
                if d <= self.max_dist:
                    pairs.append((d, bi, oi))
        pairs.sort(key=lambda x: x[0])

        used_boxes: set[int] = set()
        used_objs: set[int] = set()
        assigned_map: dict[int, int] = {}

        for _, bi, oi in pairs:
            if bi in used_boxes or oi in used_objs:
                continue
            oid = obj_ids[oi]
            assigned_map[bi] = oid
            self.objects[oid] = centroids[bi]
            self.missing[oid] = 0
            used_boxes.add(bi)
            used_objs.add(oi)

        for oi, oid in enumerate(obj_ids):
            if oi not in used_objs:
                self.missing[oid] = self.missing.get(oid, 0) + 1
                if self.missing[oid] > self.max_missing:
                    del self.missing[oid]
                    del self.objects[oid]

        assigned = []
        for bi, c in enumerate(centroids):
            if bi in assigned_map:
                assigned.append(assigned_map[bi])
            else:
                oid = self.next_id
                self.next_id += 1
                self.objects[oid] = c
                self.missing[oid] = 0
                assigned.append(oid)
        return assigned


def draw_hud_box(frame, lines, top_left=(15, 15), accent_color=(40, 220, 90)):
    x, y = top_left
    line_height = 28
    max_w = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] for line in lines)
    box_w = max_w + 30
    box_h = len(lines) * line_height + 18

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (18, 22, 30), -1)
    cv2.addWeighted(overlay, 0.74, frame, 0.26, 0, frame)
    cv2.rectangle(frame, (x, y), (x + 5, y + box_h), accent_color, -1)

    for idx, line in enumerate(lines):
        ty = y + 26 + idx * line_height
        color = accent_color if idx == 0 else (240, 244, 248)
        cv2.putText(frame, line, (x + 16, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.63, color, 2, cv2.LINE_AA)


def main(source_path: Path | None = None, headless: bool = False, max_frames: int | None = None):
    print("=" * 65)
    print(" VisionPort – Task 20: Triple-Riding Detection on Two-Wheelers")
    print("=" * 65)

    target = source_path or (SAMPLES_DIR / "traffic_sample.mp4")
    if not target.exists():
        target = SAMPLES_DIR / "traffic_sample.mp4"

    print(f"Loading YOLOv8 model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(target))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {target}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_delay = max(1, int(1000 / fps))

    tracker = CentroidTracker(max_dist=max(65, width // 11))
    violation_bike_ids: set[int] = set()
    all_bike_ids: set[int] = set()
    max_riders_seen = 0
    frame_idx = 0

    print("Processing video frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break
        frame_idx += 1

        # Detect persons (0), bicycles (1), and motorcycles (3)
        res = model.predict(frame, classes=[0, 1, 3], conf=0.25, verbose=False)[0]
        bike_boxes = []
        person_boxes = []

        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                coords = box.xyxy[0].cpu().tolist()
                if cls_id in TWO_WHEELER_CLASSES:
                    bike_boxes.append(coords)
                elif cls_id == PERSON_CLASS:
                    person_boxes.append(coords)

        bike_ids = tracker.update(bike_boxes)
        bike_riders: dict[int, list] = {i: [] for i in range(len(bike_boxes))}

        # Associate persons with nearest two-wheeler
        for pbox in person_boxes:
            px1, py1, px2, py2 = pbox
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0
            p_hip_y = py1 + (py2 - py1) * 0.65

            best_idx = None
            best_dist = float("inf")

            for bi, bbox in enumerate(bike_boxes):
                bx1, by1, bx2, by2 = bbox
                bw = max(1.0, bx2 - bx1)
                bh = max(1.0, by2 - by1)
                bcx = (bx1 + bx2) / 2.0
                bcy = (by1 + by2) / 2.0

                # Expanded rider zone above and around the two-wheeler
                rx1 = bx1 - 0.22 * bw
                rx2 = bx2 + 0.22 * bw
                ry1 = max(0.0, by1 - 0.75 * bh)
                ry2 = by2 + 0.10 * bh

                if (rx1 <= pcx <= rx2) and (ry1 <= p_hip_y <= ry2 or ry1 <= pcy <= ry2):
                    d = math.hypot(pcx - bcx, p_hip_y - bcy)
                    if d < best_dist:
                        best_dist = d
                        best_idx = bi

            if best_idx is not None:
                bike_riders[best_idx].append(pbox)

        frame_violations = 0
        for bi, (bbox, tid) in enumerate(zip(bike_boxes, bike_ids)):
            all_bike_ids.add(tid)
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            matched_persons = bike_riders[bi]
            rider_count = len(matched_persons)
            max_riders_seen = max(max_riders_seen, rider_count)

            for pbox in matched_persons:
                px1, py1, px2, py2 = [int(round(float(v))) for v in pbox]
                cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 220, 80), 1)

            if rider_count > 2:
                violation_bike_ids.add(tid)
                frame_violations += 1
                color = (0, 0, 255)
                label = f"TRIPLE RIDING! ID #{tid} ({rider_count} Persons)"
            else:
                color = (50, 230, 90)
                label = f"2-Wheeler ID #{tid} | Riders: {rider_count}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if rider_count > 2 else 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        if frame_violations > 0:
            cv2.rectangle(frame, (3, 3), (width - 4, height - 4), (0, 0, 255), 6)

        draw_hud_box(frame, [
            "VisionPort | Task 20: Triple-Riding Violation Detector",
            f"Two-Wheelers: {len(bike_boxes)} | Triple-Riding Alerts: {len(violation_bike_ids)}",
        ], accent_color=(0, 0, 255) if frame_violations > 0 else (168, 85, 247))

        if not headless:
            cv2.imshow("Task 20 - Triple Riding", frame)
            if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print("-" * 65)
    print(f"Frames processed         : {frame_idx}")
    print(f"Two-wheelers detected    : {len(all_bike_ids)}")
    print(f"Triple-riding violations : {len(violation_bike_ids)}")
    print(f"Max riders on one bike   : {max_riders_seen}")
    print(f"Status                   : {'TRIPLE-RIDING ALERT' if violation_bike_ids else 'COMPLIANT'}")
    print("=" * 65)
    print("[PASS] Task 20 completed successfully.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 20: Triple-Riding Detection")
    parser.add_argument("--source", type=str, default=None, help="Path to video file")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    src = Path(args.source) if args.source else None
    sys.exit(main(source_path=src, headless=args.headless, max_frames=args.max_frames))

