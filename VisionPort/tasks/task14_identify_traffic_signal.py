"""Task 14: Identify traffic signal status (red, yellow, green).

Task No: 14
Description: Identify traffic signal status (red, yellow, green).
Expected Output: Signal status detection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"

# COCO Traffic Light class ID = 9
TRAFFIC_LIGHT_CLASS_ID = 9


def detect_signal_color(roi_bgr: np.ndarray) -> tuple[str, float, dict]:
    """Detect active signal color (RED, YELLOW, GREEN) via HSV color space segmentation."""
    if roi_bgr is None or roi_bgr.size == 0:
        return "UNKNOWN", 0.0, {}

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_img, w_img = hsv.shape[:2]
    total_px = max(1, h_img * w_img)

    # HSV Masks for Red, Yellow, Green
    red1 = cv2.inRange(hsv, np.array([0, 90, 90]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 90, 90]), np.array([179, 255, 255]))
    red_mask = cv2.bitwise_or(red1, red2)

    yellow_mask = cv2.inRange(hsv, np.array([15, 90, 100]), np.array([35, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([40, 80, 80]), np.array([95, 255, 255]))

    # Boost score slightly when light sits in vertical third (top=red, mid=yellow, bot=green)
    third = max(1, h_img // 3)
    red_score = (cv2.countNonZero(red_mask) + 0.5 * cv2.countNonZero(red_mask[:third, :])) / total_px
    yellow_score = (cv2.countNonZero(yellow_mask) + 0.5 * cv2.countNonZero(yellow_mask[third : 2 * third, :])) / total_px
    green_score = (cv2.countNonZero(green_mask) + 0.5 * cv2.countNonZero(green_mask[2 * third :, :])) / total_px

    scores = {"RED": red_score, "YELLOW": yellow_score, "GREEN": green_score}
    best_color = max(scores, key=scores.get)
    best_val = scores[best_color]
    sum_val = sum(scores.values())

    if best_val < 0.012:
        return "UNKNOWN", 0.0, scores

    confidence = best_val / max(1e-6, sum_val)
    return best_color, round(confidence, 3), scores


def main(source_path: Path | None = None, conf_thresh: float = 0.18, headless: bool = False, max_frames: int | None = None):
    print("=" * 65)
    print(" VisionPort – Task 14: Identify Traffic Signal Status")
    print("=" * 65)

    target_source = source_path or (SAMPLES_DIR / "traffic_sample.mp4")
    if not target_source.exists():
        target_source = SAMPLES_DIR / "traffic_signal.jpg"

    print(f"Loading YOLOv8 model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    is_video = target_source.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")

    color_bgr = {
        "RED": (0, 0, 255),
        "YELLOW": (0, 220, 255),
        "GREEN": (0, 220, 0),
        "UNKNOWN": (180, 180, 180),
    }

    if not is_video:
        img = cv2.imread(str(target_source))
        if img is None:
            print(f"[ERROR] Could not read image: {target_source}")
            return 1

        annotated = img.copy()
        h, w = img.shape[:2]

        # First find traffic lights with YOLO
        results = model.predict(img, classes=[TRAFFIC_LIGHT_CLASS_ID], conf=conf_thresh, verbose=False)
        boxes = results[0].boxes

        chosen_bbox = None
        if boxes and len(boxes) > 0:
            # Pick highest confidence traffic light
            best_conf = -1.0
            for box in boxes:
                c = float(box.conf[0])
                if c > best_conf:
                    best_conf = c
                    chosen_bbox = list(map(int, box.xyxy[0]))
        else:
            # Fallback to whole image if cropped signal
            chosen_bbox = [0, 0, w, h]

        x1, y1, x2, y2 = chosen_bbox
        roi = img[y1:y2, x1:x2]
        status, conf, scores = detect_signal_color(roi)

        # Draw box and indicator
        badge_col = color_bgr.get(status, (180, 180, 180))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), badge_col, 3)
        cv2.putText(
            annotated,
            f"SIGNAL: {status} ({conf*100:.1f}%)",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            badge_col,
            2,
            cv2.LINE_AA,
        )

        # Top Banner
        cv2.rectangle(annotated, (0, 0), (w, 55), (20, 20, 25), -1)
        cv2.putText(
            annotated,
            f"TASK 14: TRAFFIC SIGNAL STATUS = {status} (Confidence: {conf*100:.1f}%)",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            badge_col,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"HSV Color Scores -> Red: {scores.get('RED',0):.3f} | Yellow: {scores.get('YELLOW',0):.3f} | Green: {scores.get('GREEN',0):.3f}",
            (16, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        print(f"Detected Signal: {status} (Confidence: {conf*100:.1f}%)")

        if not headless:
            cv2.imshow("VisionPort Task 14 - Signal Status", annotated)
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
        h, w = frame.shape[:2]
        annotated = frame.copy()

        results = model.predict(frame, classes=[TRAFFIC_LIGHT_CLASS_ID], conf=conf_thresh, verbose=False)
        boxes = results[0].boxes

        chosen_bbox = None
        if boxes and len(boxes) > 0:
            best_conf = -1.0
            for box in boxes:
                c = float(box.conf[0])
                if c > best_conf:
                    best_conf = c
                    chosen_bbox = list(map(int, box.xyxy[0]))

        if chosen_bbox is not None:
            x1, y1, x2, y2 = chosen_bbox
            roi = frame[y1:y2, x1:x2]
            status, conf, scores = detect_signal_color(roi)
            badge_col = color_bgr.get(status, (180, 180, 180))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), badge_col, 3)
            cv2.putText(
                annotated,
                f"SIGNAL: {status} ({conf*100:.0f}%)",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                badge_col,
                2,
                cv2.LINE_AA,
            )
        else:
            status, conf = "UNKNOWN", 0.0
            badge_col = (180, 180, 180)

        # Top Banner
        cv2.rectangle(annotated, (0, 0), (w, 45), (20, 20, 25), -1)
        cv2.putText(
            annotated,
            f"TASK 14: FRAME {frame_idx} | SIGNAL STATUS: {status} ({conf*100:.0f}%)",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            badge_col,
            2,
            cv2.LINE_AA,
        )



        if not headless:
            cv2.imshow("VisionPort Task 14 - Signal Status (Press Q to Quit)", annotated)
            key = cv2.waitKey(frame_delay) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print(f"Processed {frame_idx} video frames. Expected Output: Signal status detection [COMPLETED].")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 14: Identify Traffic Signal Status")
    parser.add_argument("--source", type=Path, default=None, help="Input video or image path")
    parser.add_argument("--conf", type=float, default=0.18, help="Confidence threshold")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed")
    args = parser.parse_args()

    sys.exit(main(source_path=args.source, conf_thresh=args.conf, headless=args.headless, max_frames=args.max_frames))

