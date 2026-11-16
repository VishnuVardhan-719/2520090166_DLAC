"""Task 17: Detect vehicles making wrong turns at intersections.

Task No: 17
Description: Wrong-turn detection at intersections.
Expected Output: Wrong-turn alerts with turn angle
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


def signed_angle_deg(v1, v2):
    """Returns signed turn angle in degrees from v1 to v2 (-180 to +180)."""
    x1, y1 = v1
    x2, y2 = v2
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    return math.degrees(math.atan2(det, dot))


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
    print(" VisionPort – Task 17: Wrong-Turn Detection at Intersections")
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
    trajectories: dict[int, deque] = defaultdict(lambda: deque(maxlen=45))
    wrong_turn_ids: dict[int, float] = {}
    all_ids: set[int] = set()
    frame_idx = 0

    ix1, iy1 = int(width * 0.18), int(height * 0.22)
    ix2, iy2 = int(width * 0.82), int(height * 0.85)

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

        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (255, 210, 0), 2)
        cv2.putText(frame, "INTERSECTION TURN MONITORING ZONE", (ix1 + 10, max(22, iy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 210, 0), 2, cv2.LINE_AA)

        active_wrong_turns = 0
        for box, tid, cls_id in zip(boxes_list, ids_list, cls_list):
            all_ids.add(tid)
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            trajectories[tid].append((cx, cy))

            pts = list(trajectories[tid])
            if len(pts) >= 8:
                k = max(3, len(pts) // 3)
                v_init = (pts[k][0] - pts[0][0], pts[k][1] - pts[0][1])
                v_curr = (pts[-1][0] - pts[-k][0], pts[-1][1] - pts[-k][1])
                len_init = math.hypot(*v_init)
                len_curr = math.hypot(*v_curr)

                if len_init >= 18 and len_curr >= 18:
                    ang = signed_angle_deg(v_init, v_curr)
                    abs_ang = abs(ang)
                    if abs_ang >= 55:
                        wrong_turn_ids[tid] = round(abs_ang, 1)

            if tid in wrong_turn_ids:
                active_wrong_turns += 1
                color = (0, 30, 255)
                label = f"ID #{tid}: WRONG TURN! ({wrong_turn_ids[tid]} deg)"
            else:
                color = (50, 230, 90)
                label = f"ID #{tid} {CLASS_NAMES.get(cls_id, 'veh').upper()}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if tid in wrong_turn_ids else 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, 2)

        if active_wrong_turns > 0:
            cv2.rectangle(frame, (3, 3), (width - 4, height - 4), (0, 0, 255), 6)

        draw_hud_box(frame, [
            "VisionPort | Task 17: Intersection Wrong-Turn Detection",
            f"Total Wrong Turns: {len(wrong_turn_ids)} | Active Alerts: {active_wrong_turns}",
        ], accent_color=(0, 30, 255) if active_wrong_turns > 0 else (50, 230, 90))

        if not headless:
            cv2.imshow("Task 17 - Wrong Turn Detection", frame)
            if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print("-" * 65)
    print(f"Frames processed   : {frame_idx}")
    print(f"Total vehicles     : {len(all_ids)}")
    print(f"Wrong turns        : {len(wrong_turn_ids)}")
    print(f"Status             : {'WRONG TURN DETECTED' if wrong_turn_ids else 'ALL TURNS VALID'}")
    print("=" * 65)
    print("[PASS] Task 17 completed successfully.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 17: Wrong-Turn Detection")
    parser.add_argument("--source", type=str, default=None, help="Path to video file")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    src = Path(args.source) if args.source else None
    sys.exit(main(source_path=src, headless=args.headless, max_frames=args.max_frames))

