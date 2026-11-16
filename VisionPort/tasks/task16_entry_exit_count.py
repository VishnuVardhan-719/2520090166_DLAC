"""Task 16: Count vehicles entering and leaving a traffic junction.

Task No: 16
Description: Entry and exit vehicle count at junctions.
Expected Output: Entry/exit counts and net vehicles inside junction
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


def main(source_path: Path | None = None, entry_pct: int = 38, exit_pct: int = 70,
         headless: bool = False, max_frames: int | None = None):
    print("=" * 65)
    print(" VisionPort – Task 16: Junction Entry & Exit Vehicle Count")
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

    entry_y = int(height * (entry_pct / 100.0))
    exit_y = int(height * (exit_pct / 100.0))

    tracker = CentroidTracker(max_dist=max(60, width // 12))
    prev_y: dict[int, int] = {}
    entered_ids: set[int] = set()
    exited_ids: set[int] = set()
    all_seen_ids: set[int] = set()
    frame_idx = 0

    print(f"Entry line at {entry_pct}% ({entry_y}px), Exit line at {exit_pct}% ({exit_y}px)")
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

        # Junction zone overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, entry_y), (width, exit_y), (30, 80, 110), -1)
        cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)

        for box, tid, cls_id in zip(boxes_list, ids_list, cls_list):
            all_seen_ids.add(tid)
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            if tid in prev_y:
                py = prev_y[tid]
                if ((py < entry_y <= cy) or (py > entry_y >= cy)) and tid not in entered_ids:
                    entered_ids.add(tid)
                if ((py < exit_y <= cy) or (py > exit_y >= cy)) and tid not in exited_ids:
                    exited_ids.add(tid)

            prev_y[tid] = cy

            if tid in exited_ids:
                color = (0, 165, 255)
                tag = f"ID #{tid} EXITED"
            elif tid in entered_ids:
                color = (50, 230, 90)
                tag = f"ID #{tid} ENTERED"
            else:
                color = (230, 210, 80)
                tag = f"ID #{tid} {CLASS_NAMES.get(cls_id, 'veh').upper()}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)
            cv2.putText(frame, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        cv2.line(frame, (0, entry_y), (width, entry_y), (50, 235, 90), 2)
        cv2.putText(frame, f"JUNCTION ENTRY LINE ({len(entered_ids)})", (16, max(22, entry_y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 235, 90), 2, cv2.LINE_AA)

        cv2.line(frame, (0, exit_y), (width, exit_y), (0, 165, 255), 2)
        cv2.putText(frame, f"JUNCTION EXIT LINE ({len(exited_ids)})", (16, max(22, exit_y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

        draw_hud_box(frame, [
            "VisionPort | Task 16: Junction Entry & Exit Counter",
            f"Entered: {len(entered_ids)} | Exited: {len(exited_ids)} | Active: {max(0, len(entered_ids) - len(exited_ids))}",
        ], accent_color=(50, 235, 90))

        if not headless:
            cv2.imshow("Task 16 - Entry & Exit Count", frame)
            if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print("-" * 65)
    print(f"Frames processed   : {frame_idx}")
    print(f"Total unique IDs   : {len(all_seen_ids)}")
    print(f"Vehicles entered   : {len(entered_ids)}")
    print(f"Vehicles exited    : {len(exited_ids)}")
    print(f"Net inside junction: {max(0, len(entered_ids) - len(exited_ids))}")
    print("=" * 65)
    print("[PASS] Task 16 completed successfully.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 16: Junction Entry & Exit Count")
    parser.add_argument("--source", type=str, default=None, help="Path to video file")
    parser.add_argument("--entry-pct", type=int, default=38, help="Entry line position (percentage of height)")
    parser.add_argument("--exit-pct", type=int, default=70, help="Exit line position (percentage of height)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    src = Path(args.source) if args.source else None
    sys.exit(main(source_path=src, entry_pct=args.entry_pct, exit_pct=args.exit_pct,
                  headless=args.headless, max_frames=args.max_frames))

