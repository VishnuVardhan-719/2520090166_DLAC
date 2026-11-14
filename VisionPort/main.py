"""VisionPort Live Portal - Main Application Server.

Runs the interactive web evaluation dashboard on http://127.0.0.1:8000
Supports task selection (Tasks 1 to 15 + All-in-One), file upload, webcam capture, and real-time YOLOv8 analytics.
"""

from __future__ import annotations

import base64
import importlib
import io
import os
import platform
import sys
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
DEFAULT_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"

SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI App
app = FastAPI(
    title="VisionPort – Intelligent Vehicle Detection & Traffic Monitoring",
    description="Interactive evaluation portal for Deep Learning traffic tasks 1 to 15.",
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")
app.mount("/data", StaticFiles(directory=str(SAMPLES_DIR)), name="data")

# Load YOLO model
model = YOLO(str(DEFAULT_WEIGHTS))

# Vehicle Class Constants
TRAFFIC_LIGHT_CLASS_ID = 9
CAR_CLASS_IDS = [2]
BIKE_CLASS_IDS = [1, 3]  # bicycle, motorcycle
HEAVY_VEHICLE_CLASS_IDS = [5, 7]  # bus, truck
ALL_VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]

CLASS_COLORS_BGR = {
    "car": (255, 180, 0),        # Cyan
    "bicycle": (50, 255, 150),   # Light Green
    "motorcycle": (0, 220, 0),    # Bright Green
    "bike": (0, 220, 0),
    "bus": (255, 80, 180),        # Magenta
    "truck": (0, 140, 255),       # Orange
}


class SpeedEstimator:
    """Estimates vehicle speeds in km/h from tracked centroid displacement."""

    def __init__(self, fps: float = 24.0, pixels_per_meter: float = 12.0):
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.history = defaultdict(lambda: deque(maxlen=8))
        self.smoothed = {}

    def update(self, detections: list[dict], frame_idx: int) -> dict[int, float]:
        speeds = {}
        for d in detections:
            tid = d.get("track_id")
            if tid is None:
                continue
            cx, cy = d["center"]
            hist = self.history[tid]
            hist.append((frame_idx, cx, cy))
            if len(hist) >= 2:
                f0, x0, y0 = hist[0]
                f1, x1, y1 = hist[-1]
                dt_sec = max(1, f1 - f0) / self.fps
                dist_px = float(np.hypot(x1 - x0, y1 - y0))
                dist_m = dist_px / self.pixels_per_meter
                speed_kmh = (dist_m / dt_sec) * 3.6
                if tid in self.smoothed:
                    speed_kmh = 0.6 * speed_kmh + 0.4 * self.smoothed[tid]
                self.smoothed[tid] = speed_kmh
            else:
                speed_kmh = self.smoothed.get(tid, 0.0)
            speeds[tid] = round(speed_kmh, 1)
        return speeds

    def count_stopped(self, speeds: dict[int, float]) -> int:
        return sum(1 for s in speeds.values() if s <= 3.0)


speed_estimator = SpeedEstimator(fps=24.0, pixels_per_meter=12.0)
session_frame_idx = 0


# Detection and Traffic Logic Helpers
def detect_vehicles(img: np.ndarray, class_ids: list[int], conf: float = 0.25) -> list[dict]:
    results = model.predict(img, classes=class_ids, conf=conf, verbose=False)
    dets = []
    if results[0].boxes and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clses = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        for box, cid, cf in zip(boxes, clses, confs):
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            name = results[0].names[cid]
            norm_name = "bike" if name in ("bicycle", "motorcycle") else name
            dets.append({
                "track_id": None,
                "class_id": int(cid),
                "raw_label": name,
                "label": norm_name,
                "confidence": float(cf),
                "bbox": [x1, y1, x2, y2],
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            })
    return dets


def track_vehicles_stream(img: np.ndarray, class_ids: list[int], conf: float = 0.25) -> list[dict]:
    results = model.track(img, persist=True, classes=class_ids, conf=conf, verbose=False)
    dets = []
    if results[0].boxes and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clses = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else [None] * len(boxes)
        for box, cid, cf, tid in zip(boxes, clses, confs, ids):
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            name = results[0].names[cid]
            norm_name = "bike" if name in ("bicycle", "motorcycle") else name
            dets.append({
                "track_id": int(tid) if tid is not None else None,
                "class_id": int(cid),
                "raw_label": name,
                "label": norm_name,
                "confidence": float(cf),
                "bbox": [x1, y1, x2, y2],
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            })
    return dets


def count_breakdown(dets: list[dict]) -> dict:
    counts = {"total": len(dets), "car": 0, "bike": 0, "bus": 0, "truck": 0}
    for d in dets:
        lbl = d["label"]
        if lbl in counts:
            counts[lbl] += 1
        elif lbl in ("bicycle", "motorcycle"):
            counts["bike"] += 1
    return counts


def classify_density(count: int) -> str:
    if count <= 5:
        return "LOW"
    elif count <= 12:
        return "MEDIUM"
    return "HIGH"


def detect_signal(img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    results = model.predict(img, classes=[TRAFFIC_LIGHT_CLASS_ID], conf=0.18, verbose=False)
    boxes = results[0].boxes
    chosen_bbox = None
    if boxes and len(boxes) > 0:
        best_conf = -1.0
        for b in boxes:
            cf = float(b.conf[0])
            if cf > best_conf:
                best_conf = cf
                chosen_bbox = [int(round(float(v))) for v in b.xyxy[0]]
    else:
        chosen_bbox = [0, 0, w, h]

    x1, y1, x2, y2 = chosen_bbox
    roi = img[y1:y2, x1:x2]
    if roi is None or roi.size == 0:
        return {"status": "UNKNOWN", "confidence": 0.0, "bbox": chosen_bbox, "scores": {}}

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hr, wr = hsv.shape[:2]
    total_px = max(1, hr * wr)

    red1 = cv2.inRange(hsv, np.array([0, 90, 90]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 90, 90]), np.array([179, 255, 255]))
    red_mask = cv2.bitwise_or(red1, red2)
    yellow_mask = cv2.inRange(hsv, np.array([15, 90, 100]), np.array([35, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([40, 80, 80]), np.array([95, 255, 255]))

    third = max(1, hr // 3)
    red_score = (cv2.countNonZero(red_mask) + 0.5 * cv2.countNonZero(red_mask[:third, :])) / total_px
    yellow_score = (cv2.countNonZero(yellow_mask) + 0.5 * cv2.countNonZero(yellow_mask[third : 2 * third, :])) / total_px
    green_score = (cv2.countNonZero(green_mask) + 0.5 * cv2.countNonZero(green_mask[2 * third :, :])) / total_px

    scores = {"RED": red_score, "YELLOW": yellow_score, "GREEN": green_score}
    best = max(scores, key=scores.get)
    best_val = scores[best]

    if best_val < 0.012:
        return {"status": "UNKNOWN", "confidence": 0.0, "bbox": chosen_bbox, "scores": scores}

    conf = best_val / max(1e-6, sum(scores.values()))
    return {"status": best, "confidence": round(float(conf), 3), "bbox": chosen_bbox, "scores": scores}


def estimate_waiting(count: int, signal: str = "RED") -> dict:
    clearance_sec = count / 1.0  # 1 vehicle per sec across 2 lanes
    cycles = max(1, int(np.ceil(count / 15.0))) if count > 0 else 0
    if count == 0:
        wait_sec = 0.0
    elif signal == "RED":
        wait_sec = 30.0 + clearance_sec + max(0, cycles - 1) * 45.0
    elif signal == "YELLOW":
        wait_sec = 45.0 + clearance_sec
    elif signal == "GREEN":
        wait_sec = clearance_sec * 0.35
    else:
        wait_sec = clearance_sec + 20.0

    return {
        "vehicle_count": count,
        "signal_status": signal,
        "queue_clearance_sec": round(clearance_sec, 1),
        "cycles_needed": cycles,
        "estimated_wait_sec": round(wait_sec, 1),
    }


def draw_hud(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    annotated = frame.copy()
    if not lines:
        return annotated
    h, w = annotated.shape[:2]
    panel_h = 16 + 28 * len(lines)
    panel_w = min(w - 20, 560)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.72, annotated, 0.28, 0, annotated)

    for idx, (text, color) in enumerate(lines):
        y = 35 + idx * 28
        cv2.putText(annotated, text, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    return annotated


def draw_boxes(frame: np.ndarray, dets: list[dict], show_conf: bool = True, show_id: bool = False, speeds: dict = None) -> np.ndarray:
    annotated = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        lbl = d["label"]
        col = CLASS_COLORS_BGR.get(lbl, (0, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)

        parts = []
        if show_id and d.get("track_id") is not None:
            parts.append(f"ID:{d['track_id']}")
        parts.append(lbl.upper())
        if show_conf:
            parts.append(f"{d['confidence']*100:.0f}%")
        if speeds and d.get("track_id") in speeds:
            parts.append(f"{speeds[d['track_id']]:.1f}km/h")

        text = " | ".join(parts)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(th + 6, y1)
        cv2.rectangle(annotated, (x1, ty - th - 6), (x1 + tw + 6, ty + 2), col, -1)
        cv2.putText(annotated, text, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return annotated


# Task Catalog
TASKS = {
    1: {"id": 1, "title": "Install Python, OpenCV & Required Libraries", "expected": "Working Python environment", "script": "task1_setup_and_verify.py"},
    2: {"id": 2, "title": "Capture Image Using Camera", "expected": "Traffic image capture", "script": "task2_capture_traffic_signal.py"},
    3: {"id": 3, "title": "Read & Display Traffic Images", "expected": "Image display window", "script": "task3_read_display_traffic_image.py"},
    4: {"id": 4, "title": "Load & Play Traffic Video", "expected": "Video playback", "script": "task4_load_play_traffic_video.py"},
    5: {"id": 5, "title": "Extract Frames from Traffic Videos", "expected": "Frames saved as images", "script": "task5_extract_video_frames.py"},
    6: {"id": 6, "title": "Detect Cars Using YOLO", "expected": "Car detection with bounding boxes", "script": "task6_detect_cars.py"},
    7: {"id": 7, "title": "Detect Bikes, Buses & Trucks", "expected": "Multiple vehicle detection", "script": "task7_detect_multiple_vehicles.py"},
    8: {"id": 8, "title": "Display Vehicle Names & Confidence Scores", "expected": "Labelled detection output", "script": "task8_display_labels_confidence.py"},
    9: {"id": 9, "title": "Count Number of Vehicles in Frame", "expected": "Vehicle count", "script": "task9_count_vehicles.py"},
    10: {"id": 10, "title": "Calculate Traffic Density (Low/Med/High)", "expected": "Low, medium and high traffic", "script": "task10_traffic_density.py"},
    11: {"id": 11, "title": "Detect Congestion at Signal Junctions", "expected": "Congestion alerts", "script": "task11_detect_congestion.py"},
    12: {"id": 12, "title": "Track Individual Vehicles Across Frames", "expected": "Vehicle tracking IDs", "script": "task12_track_vehicles.py"},
    13: {"id": 13, "title": "Calculate Approximate Vehicle Speed (km/h)", "expected": "Speed estimation", "script": "task13_calculate_vehicle_speed.py"},
    14: {"id": 14, "title": "Identify Traffic Signal Status (Red/Yellow/Green)", "expected": "Signal status detection", "script": "task14_identify_traffic_signal.py"},
    15: {"id": 15, "title": "Estimate Traffic Waiting Time", "expected": "Waiting-time report", "script": "task15_estimate_waiting_time.py"},
    "all": {"id": "all", "title": "Full Telemetry Pipeline (All Tasks Combined)", "expected": "Unified Traffic Intelligence Dashboard", "script": "run_all_tasks.py"},
}

SAMPLE_PRESETS = [
    {"id": "intersection", "label": "🚌 Urban Intersection", "filename": "traffic_intersection.jpg", "type": "image"},
    {"id": "congested", "label": "🚨 Heavy Congestion", "filename": "traffic_congested.jpg", "type": "image"},
    {"id": "signal", "label": "🚦 Traffic Signal", "filename": "traffic_signal.jpg", "type": "image"},
    {"id": "video_sample", "label": "🎥 Traffic Video", "filename": "traffic_sample.mp4", "type": "video"},
]


def check_python_environment() -> dict:
    modules_to_check = [
        ("cv2", "OpenCV", "5.0.0"),
        ("numpy", "NumPy", "2.5.4"),
        ("torch", "PyTorch", "2.15.0"),
        ("torchvision", "TorchVision", "0.29.2"),
        ("ultralytics", "Ultralytics YOLO", "8.4.211"),
        ("pandas", "Pandas", "3.1.1"),
        ("matplotlib", "Matplotlib", "3.11.2"),
        ("fastapi", "FastAPI", "0.162.5"),
        ("uvicorn", "Uvicorn", "0.54.0"),
        ("lap", "Linear Assignment (LAP Tracker)", "0.5.14"),
    ]
    report = []
    all_ok = True
    for mod_name, label, _ in modules_to_check:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "Installed")
            report.append({"name": label, "module": mod_name, "version": ver, "status": "OK"})
        except Exception as exc:
            all_ok = False
            report.append({"name": label, "module": mod_name, "version": str(exc), "status": "FAIL"})

    import torch

    has_cuda = torch.cuda.is_available()
    weights_ok = DEFAULT_WEIGHTS.exists()
    weights_mb = round(DEFAULT_WEIGHTS.stat().st_size / (1024 * 1024), 2) if weights_ok else 0

    return {
        "status": "WORKING" if (all_ok and weights_ok) else "INCOMPLETE",
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "os_platform": f"{platform.system()} {platform.release()}",
        "cuda_available": has_cuda,
        "hardware_accelerator": "CUDA" if has_cuda else "CPU",
        "weights_present": weights_ok,
        "weights_size_mb": weights_mb,
        "libraries": report,
    }


def execute_task_pipeline(task_id: int | str, img_bgr: np.ndarray, conf_thresh: float = 0.25) -> tuple[np.ndarray, dict]:
    global session_frame_idx
    session_frame_idx += 1
    t0 = time.perf_counter()
    h, w = img_bgr.shape[:2]
    annotated = img_bgr.copy()

    t_key = "all" if str(task_id).lower() in ("all", "0") else int(task_id)
    info = TASKS.get(t_key, TASKS["all"])

    telemetry = {
        "task_id": t_key,
        "task_title": info["title"],
        "expected_output": info["expected"],
        "total_vehicles": 0,
        "breakdown": {"car": 0, "bike": 0, "bus": 0, "truck": 0},
        "density": "LOW",
        "congested": False,
        "congestion_severity": "NONE",
        "congestion_reason": "Normal traffic flow",
        "signal_status": "UNKNOWN",
        "signal_confidence": 0.0,
        "avg_speed_kmh": 0.0,
        "wait_time_sec": 0.0,
        "queue_clearance_sec": 0.0,
        "detections": [],
        "details": {},
    }

    if t_key == 1:
        env = check_python_environment()
        telemetry["details"] = env
        annotated = draw_hud(
            annotated,
            [
                (f"TASK 1: ENVIRONMENT = {env['status']}", (0, 255, 0) if env["status"] == "WORKING" else (0, 0, 255)),
                (f"Python {env['python_version']} | OpenCV {cv2.__version__}", (230, 230, 230)),
                (f"YOLOv8 Weights: yolov8n.pt ({env['weights_size_mb']} MB)", (0, 255, 255)),
            ],
        )

    elif t_key == 2:
        sig = detect_signal(img_bgr)
        telemetry["signal_status"] = sig["status"]
        telemetry["signal_confidence"] = sig["confidence"]
        annotated = draw_hud(
            annotated,
            [
                ("TASK 2: TRAFFIC SIGNAL CAMERA CAPTURE", (0, 255, 0)),
                (f"Capture Time: {time.strftime('%H:%M:%S')} | Res: {w}x{h}", (230, 230, 230)),
                (f"Signal: {sig['status']} ({sig['confidence']*100:.0f}%)", (0, 255, 255)),
            ],
        )

    elif t_key == 3:
        file_size_kb = round(img_bgr.nbytes / 1024, 1)
        aspect = round(w / max(1, h), 2)
        annotated = draw_hud(
            annotated,
            [
                (f"TASK 3: READ & DISPLAY IMAGE ({w}x{h}, {aspect}:1)", (0, 255, 255)),
                (f"Buffer: {file_size_kb} KB | Range: [{img_bgr.min()}-{img_bgr.max()}]", (220, 220, 220)),
            ],
        )

    elif t_key == 6:
        dets = detect_vehicles(img_bgr, class_ids=CAR_CLASS_IDS, conf=conf_thresh)
        telemetry["total_vehicles"] = len(dets)
        telemetry["breakdown"]["car"] = len(dets)
        telemetry["detections"] = dets
        annotated = draw_boxes(annotated, dets, show_conf=True)
        annotated = draw_hud(
            annotated,
            [(f"TASK 6: CAR DETECTION (YOLOv8) | Cars = {len(dets)}", (0, 255, 255))],
        )

    elif t_key == 7:
        dets = detect_vehicles(img_bgr, class_ids=BIKE_CLASS_IDS + HEAVY_VEHICLE_CLASS_IDS, conf=conf_thresh)
        counts = count_breakdown(dets)
        telemetry["total_vehicles"] = len(dets)
        telemetry["breakdown"] = counts
        telemetry["detections"] = dets
        annotated = draw_boxes(annotated, dets, show_conf=True)
        annotated = draw_hud(
            annotated,
            [(f"TASK 7: MULTI-VEHICLE | Bikes:{counts['bike']} Buses:{counts['bus']} Trucks:{counts['truck']}", (255, 100, 220))],
        )

    elif t_key == 8:
        dets = detect_vehicles(img_bgr, class_ids=ALL_VEHICLE_CLASS_IDS, conf=conf_thresh)
        telemetry["total_vehicles"] = len(dets)
        telemetry["detections"] = dets
        annotated = draw_boxes(annotated, dets, show_conf=True)
        annotated = draw_hud(
            annotated,
            [(f"TASK 8: LABELLED OUTPUT | Detections: {len(dets)}", (255, 220, 0))],
        )

    elif t_key == 9:
        dets = detect_vehicles(img_bgr, class_ids=ALL_VEHICLE_CLASS_IDS, conf=conf_thresh)
        counts = count_breakdown(dets)
        telemetry["total_vehicles"] = counts["total"]
        telemetry["breakdown"] = counts
        telemetry["detections"] = dets
        annotated = draw_boxes(annotated, dets, show_conf=False)
        annotated = draw_hud(
            annotated,
            [
                (f"TASK 9: VEHICLE COUNT = {counts['total']}", (0, 255, 180)),
                (f"Cars:{counts['car']} Bikes:{counts['bike']} Buses:{counts['bus']} Trucks:{counts['truck']}", (230, 230, 230)),
            ],
        )

    elif t_key == 10:
        dets = detect_vehicles(img_bgr, class_ids=ALL_VEHICLE_CLASS_IDS, conf=conf_thresh)
        count = len(dets)
        density = classify_density(count)
        telemetry["total_vehicles"] = count
        telemetry["density"] = density
        telemetry["detections"] = dets
        col = {"LOW": (0, 255, 0), "MEDIUM": (0, 215, 255), "HIGH": (0, 0, 255)}[density]
        annotated = draw_boxes(annotated, dets, show_conf=False)
        annotated = draw_hud(
            annotated,
            [
                (f"TASK 10: TRAFFIC DENSITY = {density}", col),
                (f"Count: {count} (LOW <=5, MED <=12, HIGH >12)", (230, 230, 230)),
            ],
        )

    elif t_key == 11:
        dets = detect_vehicles(img_bgr, class_ids=ALL_VEHICLE_CLASS_IDS, conf=conf_thresh)
        count = len(dets)
        is_congested = count >= 8
        telemetry["total_vehicles"] = count
        telemetry["congested"] = is_congested
        telemetry["congestion_severity"] = "SEVERE" if count >= 12 else ("MODERATE" if is_congested else "NONE")
        telemetry["congestion_reason"] = f"Queue bottleneck ({count} vehicles >= 8)" if is_congested else "Normal flow"
        telemetry["detections"] = dets

        annotated = draw_boxes(annotated, dets, show_conf=False)
        alert_col = (0, 0, 255) if is_congested else (0, 255, 0)
        annotated = draw_hud(
            annotated,
            [
                (f"TASK 11: {'CONGESTION ALERT' if is_congested else 'JUNCTION CLEAR'}", alert_col),
                (f"Queue: {count} vehicles | Reason: {telemetry['congestion_reason']}", (230, 230, 230)),
            ],
        )
        if is_congested:
            cv2.rectangle(annotated, (4, 4), (w - 5, h - 5), (0, 0, 255), 4)

    elif t_key == 12:
        dets = track_vehicles_stream(img_bgr, class_ids=ALL_VEHICLE_CLASS_IDS, conf=conf_thresh)
        telemetry["total_vehicles"] = len(dets)
        telemetry["detections"] = dets
        annotated = draw_boxes(annotated, dets, show_conf=False, show_id=True)
        annotated = draw_hud(
            annotated,
            [(f"TASK 12: MULTI-OBJECT TRACKING | Tracked: {len(dets)}", (0, 255, 255))],
        )

    elif t_key == 13:
        dets = track_vehicles_stream(img_bgr, class_ids=ALL_VEHICLE_CLASS_IDS, conf=conf_thresh)
        speeds = speed_estimator.update(dets, session_frame_idx)
        avg_spd = sum(speeds.values()) / len(speeds) if speeds else 0.0
        telemetry["total_vehicles"] = len(dets)
        telemetry["avg_speed_kmh"] = round(avg_spd, 1)
        telemetry["detections"] = dets
        annotated = draw_boxes(annotated, dets, show_conf=False, show_id=True, speeds=speeds)
        annotated = draw_hud(
            annotated,
            [(f"TASK 13: SPEED ESTIMATION | Avg: {avg_spd:.1f} km/h", (0, 255, 180))],
        )

    elif t_key == 14:
        sig = detect_signal(img_bgr)
        telemetry["signal_status"] = sig["status"]
        telemetry["signal_confidence"] = sig["confidence"]
        col = {"RED": (0, 0, 255), "YELLOW": (0, 220, 255), "GREEN": (0, 220, 0)}.get(sig["status"], (180, 180, 180))
        x1, y1, x2, y2 = sig["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 3)
        annotated = draw_hud(
            annotated,
            [(f"TASK 14: SIGNAL = {sig['status']} ({sig['confidence']*100:.0f}%)", col)],
        )

    else:
        # Task 15 & All-in-One Full Pipeline
        dets = track_vehicles_stream(img_bgr, class_ids=ALL_VEHICLE_CLASS_IDS, conf=conf_thresh)
        counts = count_breakdown(dets)
        speeds = speed_estimator.update(dets, session_frame_idx)
        avg_spd = sum(speeds.values()) / len(speeds) if speeds else 0.0
        sig = detect_signal(img_bgr)
        density = classify_density(counts["total"])
        is_congested = counts["total"] >= 8
        wait_info = estimate_waiting(counts["total"], signal=sig["status"])

        telemetry.update({
            "total_vehicles": counts["total"],
            "breakdown": counts,
            "density": density,
            "congested": is_congested,
            "congestion_severity": "SEVERE" if counts["total"] >= 12 else ("MODERATE" if is_congested else "NONE"),
            "congestion_reason": f"Queue bottleneck ({counts['total']} vehicles)" if is_congested else "Normal flow",
            "signal_status": sig["status"],
            "signal_confidence": sig["confidence"],
            "avg_speed_kmh": round(avg_spd, 1),
            "wait_time_sec": wait_info["estimated_wait_sec"],
            "queue_clearance_sec": wait_info["queue_clearance_sec"],
            "cycles_needed": wait_info["cycles_needed"],
            "detections": dets,
        })

        annotated = draw_boxes(annotated, dets, show_conf=True, show_id=True, speeds=speeds)
        col = {"RED": (0, 0, 255), "YELLOW": (0, 220, 255), "GREEN": (0, 220, 0)}.get(sig["status"], (180, 180, 180))
        annotated = draw_hud(
            annotated,
            [
                (f"VisionPort Live Telemetry | Vehicles: {counts['total']} | Density: {density}", (0, 255, 255)),
                (f"Signal: {sig['status']} | Congestion: {telemetry['congestion_severity']} | Avg: {avg_spd:.1f} km/h", col),
                (f"Est. Wait Time: {wait_info['estimated_wait_sec']}s (Clearance: {wait_info['queue_clearance_sec']}s)", (255, 255, 255)),
            ],
        )
        if is_congested:
            cv2.rectangle(annotated, (4, 4), (w - 5, h - 5), (0, 0, 255), 4)

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    telemetry["latency_ms"] = latency_ms
    return annotated, telemetry


# Routes
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_file = TEMPLATES_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>VisionPort Portal Initializing...</h1>")


@app.get("/tasks")
async def get_tasks():
    return {"tasks": list(TASKS.values())}


@app.get("/samples_list")
async def get_samples():
    return {"samples": SAMPLE_PRESETS}


@app.get("/env_report")
async def get_env_report():
    return check_python_environment()


@app.get("/camera/capture")
async def camera_capture(task_id: str = "2"):
    cap = cv2.VideoCapture(0)
    frame = None
    if cap.isOpened():
        try:
            for _ in range(4):
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
        finally:
            cap.release()

    if frame is None:
        sample_path = SAMPLES_DIR / "traffic_signal.jpg"
        if not sample_path.exists():
            sample_path = SAMPLES_DIR / "traffic_intersection.jpg"
        frame = cv2.imread(str(sample_path))

    annotated, telemetry = execute_task_pipeline(task_id, frame)
    _, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64_str = base64.b64encode(encoded.tobytes()).decode("utf-8")

    return {
        "status": "success",
        "source": "camera",
        "telemetry": telemetry,
        "image_base64": f"data:image/jpeg;base64,{b64_str}",
    }


@app.get("/sample/{sample_name}")
async def process_sample(sample_name: str, task_id: str = "all", conf: float = 0.25):
    target = None
    for p in SAMPLE_PRESETS:
        if p["id"].lower() == sample_name.lower() or p["filename"].lower() == sample_name.lower():
            target = SAMPLES_DIR / p["filename"]
            break

    if target is None or not target.exists():
        name_lower = sample_name.lower()
        if "signal" in name_lower:
            target = SAMPLES_DIR / "traffic_signal.jpg"
        elif "congest" in name_lower:
            target = SAMPLES_DIR / "traffic_congested.jpg"
        elif "video" in name_lower or "sample" in name_lower:
            target = SAMPLES_DIR / "traffic_sample.mp4"
        else:
            target = SAMPLES_DIR / "traffic_intersection.jpg"

    if target.suffix.lower() in (".mp4", ".avi", ".mov"):
        cap = cv2.VideoCapture(str(target))
        ok, frame = cap.read()
        cap.release()
    else:
        frame = cv2.imread(str(target))

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Failed to decode sample"})

    annotated, telemetry = execute_task_pipeline(task_id, frame, conf_thresh=conf)
    _, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64_str = base64.b64encode(encoded.tobytes()).decode("utf-8")

    return {
        "status": "success",
        "sample": sample_name,
        "telemetry": telemetry,
        "image_base64": f"data:image/jpeg;base64,{b64_str}",
    }


@app.post("/process/")
async def process_file(
    file: UploadFile = File(...),
    task_id: str = Form("all"),
    conf: float = Form(0.25),
):
    contents = await file.read()
    filename = (file.filename or "upload.jpg").lower()

    if filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_path.write_bytes(contents)

        try:
            cap = cv2.VideoCapture(str(tmp_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, frame = cap.read()
            cap.release()
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        if not ok or frame is None:
            return JSONResponse(status_code=400, content={"error": "Could not decode video frame"})

        annotated, telemetry = execute_task_pipeline(task_id, frame, conf_thresh=conf)
        telemetry["video_meta"] = {
            "fps": round(fps, 1),
            "total_frames": total_frames,
            "duration_sec": round(total_frames / max(1.0, fps), 1),
        }
        _, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64_str = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return {
            "status": "success",
            "file_type": "video",
            "telemetry": telemetry,
            "image_base64": f"data:image/jpeg;base64,{b64_str}",
        }

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image format"})

    annotated, telemetry = execute_task_pipeline(task_id, img, conf_thresh=conf)
    _, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64_str = base64.b64encode(encoded.tobytes()).decode("utf-8")

    return {
        "status": "success",
        "file_type": "image",
        "telemetry": telemetry,
        "image_base64": f"data:image/jpeg;base64,{b64_str}",
    }


@app.post("/extract_frames/")
async def extract_frames_endpoint(
    sample_video: Optional[str] = Form(None),
    interval: int = Form(15),
    max_frames: int = Form(12),
):
    target = SAMPLES_DIR / "traffic_sample.mp4"
    if sample_video:
        candidate = SAMPLES_DIR / sample_video
        if candidate.exists():
            target = candidate

    cap = cv2.VideoCapture(str(target))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    extracted = []
    fidx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if fidx % max(1, interval) == 0:
            _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64_frame = base64.b64encode(encoded.tobytes()).decode("utf-8")
            extracted.append({
                "frame_index": fidx,
                "timestamp_sec": round(fidx / max(1.0, fps), 2),
                "url": f"data:image/jpeg;base64,{b64_frame}",
                "resolution": f"{w}x{h}",
            })
            if len(extracted) >= max_frames:
                break
        fidx += 1
    cap.release()

    return {"status": "success", "task_id": 5, "extracted_count": len(extracted), "frames": extracted}


@app.get("/status/")
async def system_status():
    import torch
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    return {
        "system_status": "Active",
        "app_title": "VisionPort Intelligent Traffic Monitor",
        "model_architecture": "YOLOv8 Nano (Ultralytics)",
        "hardware_accelerator": device,
        "tasks_supported": len(TASKS),
        "samples_available": len(SAMPLE_PRESETS),
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting VisionPort Live Portal on http://127.0.0.1:8000 ...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
