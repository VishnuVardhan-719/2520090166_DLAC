"""Task 18: Identify vehicles moving in the wrong direction.

Task No: 18
Description: Wrong-way vehicle alerts.
Expected Output: Wrong-way vehicle detection and flow direction
"""

from __future__ import annotations

import argparse
import math
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
CLASS_NAMES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

FLOW_VECTORS = {
    "down": (0.0, 1.0, "TOP -> BOTTOM"),
    "up": (0.0, -1.0, "BOTTOM -> TOP"),
    "right": (1.0, 0.0, "LEFT -> RIGHT"),
    "left": (-1.0, 0.0, "RIGHT -> LEFT"),
}


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


def main(source_path: Path | None = None, flow_dir: str = "auto",
         headless: bool = False, max_frames: int | None = None):
    print("=" * 65)
    print(" VisionPort – Task 18: Wrong-Way Vehicle Alerts")
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
    history: dict[int, deque] = defaultdict(lambda: deque(maxlen=15))
    wrong_way_streak: dict[int, int] = defaultdict(int)
    wrong_way_ids: set[int] = set()
    dominant_accum = [0.0, 0.0]
    frame_idx = 0

    print(f"Flow direction: {flow_dir}")
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

        # Determine allowed flow vector
        if flow_dir in FLOW_VECTORS:
            fx, fy, flow_label = FLOW_VECTORS[flow_dir]
        else:
            mag = math.hypot(dominant_accum[0], dominant_accum[1])
            if mag > 10.0:
                fx, fy = dominant_accum[0] / mag, dominant_accum[1] / mag
                if abs(fy) >= abs(fx):
                    flow_label = "AUTO: TOP -> BOTTOM" if fy > 0 else "AUTO: BOTTOM -> TOP"
                else:
                    flow_label = "AUTO: LEFT -> RIGHT" if fx > 0 else "AUTO: RIGHT -> LEFT"
            else:
                fx, fy, flow_label = 0.0, 1.0, "AUTO: CALIBRATING..."

        active_wrong_way = 0
        for box, tid, cls_id in zip(boxes_list, ids_list, cls_list):
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            history[tid].append((cx, cy))

            pts = list(history[tid])
            if len(pts) >= 4:
                dx = pts[-1][0] - pts[0][0]
                dy = pts[-1][1] - pts[0][1]
                dist = math.hypot(dx, dy)
                if dist >= 14.0:
                    if flow_dir == "auto":
                        dominant_accum[0] += dx
                        dominant_accum[1] += dy
                    proj = (dx / dist) * fx + (dy / dist) * fy
                    if proj < -0.55:
                        wrong_way_streak[tid] += 1
                        if wrong_way_streak[tid] >= 3:
                            wrong_way_ids.add(tid)
                    else:
                        wrong_way_streak[tid] = max(0, wrong_way_streak[tid] - 1)

            if tid in wrong_way_ids:
                active_wrong_way += 1
                color = (0, 0, 255)
                label = f"ALERT! ID #{tid} WRONG WAY"
            else:
                color = (50, 230, 90)
                label = f"ID #{tid} {CLASS_NAMES.get(cls_id, 'veh').upper()} (OK)"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if tid in wrong_way_ids else 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # Allowed flow arrow indicator
        ax, ay = width - 70, 65
        end_x = int(ax + fx * 32)
        end_y = int(ay + fy * 32)
        cv2.circle(frame, (ax, ay), 40, (18, 22, 30), -1)
        cv2.arrowedLine(frame, (int(ax - fx * 22), int(ay - fy * 22)), (end_x, end_y),
                        (50, 235, 90), 3, tipLength=0.35)

        if active_wrong_way > 0:
            cv2.rectangle(frame, (3, 3), (width - 4, height - 4), (0, 0, 255), 7)

        draw_hud_box(frame, [
            "VisionPort | Task 18: Wrong-Way Vehicle Detector",
            f"Allowed Flow: {flow_label}",
            f"Wrong-Way Alerts: {len(wrong_way_ids)} | Active Now: {active_wrong_way}",
        ], accent_color=(0, 0, 255) if active_wrong_way > 0 else (50, 230, 90))

        if not headless:
            cv2.imshow("Task 18 - Wrong-Way Alerts", frame)
            if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print("-" * 65)
    print(f"Frames processed   : {frame_idx}")
    print(f"Flow direction     : {flow_label}")
    print(f"Wrong-way vehicles : {len(wrong_way_ids)}")
    print(f"Status             : {'WRONG-WAY ALERT' if wrong_way_ids else 'NORMAL FLOW'}")
    print("=" * 65)
    print("[PASS] Task 18 completed successfully.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 18: Wrong-Way Vehicle Alerts")
    parser.add_argument("--source", type=str, default=None, help="Path to video file")
    parser.add_argument("--flow-dir", type=str, default="auto", choices=["auto", "up", "down", "left", "right"])
    parser.add_argument("--headless", action="store_true", help="Run without GUI display")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    src = Path(args.source) if args.source else None
    sys.exit(main(source_path=src, flow_dir=args.flow_dir, headless=args.headless, max_frames=args.max_frames))

