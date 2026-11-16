"""Task 19: Detect vehicles parked in restricted traffic areas.

Task No: 19
Description: Illegal parking detection.
Expected Output: Illegal parking violation alerts with dwell time
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

VEHICLE_CLASSES = [1, 2, 3, 5, 7]
CLASS_NAMES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


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


def get_zone_rect(zone_preset: str, width: int, height: int):
    if zone_preset == "left_curb":
        return (int(width * 0.03), int(height * 0.42), int(width * 0.42), int(height * 0.94))
    if zone_preset == "center_box":
        return (int(width * 0.25), int(height * 0.35), int(width * 0.75), int(height * 0.85))
    if zone_preset == "full_road":
        return (int(width * 0.04), int(height * 0.20), int(width * 0.96), int(height * 0.95))
    # Default: right_curb
    return (int(width * 0.56), int(height * 0.42), int(width * 0.97), int(height * 0.94))


def main(source_path: Path | None = None, zone_preset: str = "right_curb", dwell_sec: float = 1.5,
         headless: bool = False, max_frames: int | None = None):
    print("=" * 65)
    print(" VisionPort – Task 19: Illegal Parking Detection")
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

    zx1, zy1, zx2, zy2 = get_zone_rect(zone_preset, width, height)
    tracker = CentroidTracker(max_dist=max(55, width // 14))
    anchor_pos: dict[int, tuple[int, int]] = {}
    stationary_frames: dict[int, int] = defaultdict(int)
    illegally_parked_ids: set[int] = set()
    frame_idx = 0

    print(f"No-parking zone: {zone_preset} | Dwell threshold: {dwell_sec}s")
    print("Processing video frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break
        frame_idx += 1

        res = model.predict(frame, classes=VEHICLE_CLASSES, conf=0.25, verbose=False)[0]
        boxes_list, cls_list = [], []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes_list = res.boxes.xyxy.cpu().tolist()
            cls_list = res.boxes.cls.int().cpu().tolist()

        ids_list = tracker.update(boxes_list)

        # Draw no-parking zone
        overlay = frame.copy()
        cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 0, 220), -1)
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 40, 255), 3)
        cv2.putText(frame, "RESTRICTED NO-PARKING ZONE", (zx1 + 10, max(24, zy1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 60, 255), 2, cv2.LINE_AA)

        active_violations = 0
        for box, tid, cls_id in zip(boxes_list, ids_list, cls_list):
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            wheel_y = int(y1 + (y2 - y1) * 0.80)

            inside_zone = (zx1 <= cx <= zx2) and (zy1 <= wheel_y <= zy2)
            if inside_zone:
                if tid in anchor_pos:
                    ax, ay = anchor_pos[tid]
                    if math.hypot(cx - ax, wheel_y - ay) <= 10.0:
                        stationary_frames[tid] += 1
                    else:
                        anchor_pos[tid] = (cx, wheel_y)
                        stationary_frames[tid] = max(0, stationary_frames[tid] - 2)
                else:
                    anchor_pos[tid] = (cx, wheel_y)
                    stationary_frames[tid] = 1
            else:
                anchor_pos.pop(tid, None)
                stationary_frames[tid] = 0

            stopped_sec = stationary_frames[tid] / fps
            if inside_zone and stopped_sec >= dwell_sec:
                illegally_parked_ids.add(tid)

            if tid in illegally_parked_ids and inside_zone:
                active_violations += 1
                color = (0, 0, 255)
                label = f"ILLEGAL PARKING! ID #{tid} ({stopped_sec:.1f}s)"
            elif inside_zone:
                color = (0, 200, 255)
                label = f"IN ZONE ID #{tid} ({stopped_sec:.1f}s)"
            else:
                color = (50, 230, 90)
                label = f"ID #{tid} {CLASS_NAMES.get(cls_id, 'veh').upper()}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                          3 if (tid in illegally_parked_ids and inside_zone) else 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.53, color, 2, cv2.LINE_AA)

        draw_hud_box(frame, [
            "VisionPort | Task 19: Illegal Parking Detection",
            f"Violations: {len(illegally_parked_ids)} | Active in Zone: {active_violations}",
        ], accent_color=(0, 0, 255) if active_violations > 0 else (245, 158, 11))

        if not headless:
            cv2.imshow("Task 19 - Illegal Parking", frame)
            if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print("-" * 65)
    print(f"Frames processed      : {frame_idx}")
    print(f"Dwell threshold       : {dwell_sec}s")
    print(f"Illegal parking count : {len(illegally_parked_ids)}")
    print(f"Status                : {'VIOLATION DETECTED' if illegally_parked_ids else 'NO VIOLATIONS'}")
    print("=" * 65)
    print("[PASS] Task 19 completed successfully.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 19: Illegal Parking Detection")
    parser.add_argument("--source", type=str, default=None, help="Path to video file")
    parser.add_argument("--zone", type=str, default="right_curb",
                        choices=["left_curb", "right_curb", "center_box", "full_road"])
    parser.add_argument("--dwell-sec", type=float, default=1.5, help="Seconds before flagging illegal parking")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    src = Path(args.source) if args.source else None
    sys.exit(main(source_path=src, zone_preset=args.zone, dwell_sec=args.dwell_sec,
                  headless=args.headless, max_frames=args.max_frames))

