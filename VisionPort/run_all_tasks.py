"""Run and verify all 15 VisionPort individual task programs end-to-end."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TASKS_DIR = PROJECT_ROOT / "tasks"

TASKS = [
    (1, "Environment verification", "task1_setup_and_verify.py", []),
    (2, "Camera traffic signal capture", "task2_capture_traffic_signal.py", ["--auto-capture", "--headless"]),
    (3, "Read & display traffic image", "task3_read_display_traffic_image.py", ["--headless"]),
    (4, "Load & play traffic video", "task4_load_play_traffic_video.py", ["--headless", "--max-frames", "30"]),
    (5, "Extract video frames", "task5_extract_video_frames.py", ["--headless", "--interval", "15", "--max-saved", "5"]),
    (6, "Detect cars using YOLO", "task6_detect_cars.py", ["--headless", "--max-frames", "30"]),
    (7, "Detect bikes, buses & trucks", "task7_detect_multiple_vehicles.py", ["--headless", "--max-frames", "30"]),
    (8, "Display labels & confidence", "task8_display_labels_confidence.py", ["--headless", "--max-frames", "30"]),
    (9, "Count vehicles in frame", "task9_count_vehicles.py", ["--headless", "--max-frames", "30"]),
    (10, "Calculate traffic density", "task10_traffic_density.py", ["--headless", "--max-frames", "30"]),
    (11, "Detect congestion alerts", "task11_detect_congestion.py", ["--headless", "--max-frames", "30"]),
    (12, "Track vehicles across frames", "task12_track_vehicles.py", ["--headless", "--max-frames", "30"]),
    (13, "Calculate vehicle speed", "task13_calculate_vehicle_speed.py", ["--headless", "--max-frames", "30"]),
    (14, "Identify traffic signal status", "task14_identify_traffic_signal.py", ["--headless", "--max-frames", "30"]),
    (15, "Estimate traffic waiting time", "task15_estimate_waiting_time.py", ["--headless", "--max-frames", "30"]),
]


def main():
    parser = argparse.ArgumentParser(description="Run all 15 VisionPort task programs.")
    args = parser.parse_args()

    failures = []
    start_time = time.perf_counter()

    for num, title, script_name, extra_args in TASKS:
        script_path = TASKS_DIR / script_name
        cmd = [sys.executable, str(script_path), *extra_args]
        print(f"\n>>> Running Task {num}: {title} ({script_name}) ...")
        t0 = time.perf_counter()
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        dur = time.perf_counter() - t0

        if res.returncode != 0:
            failures.append((num, title, res.returncode))
            print(f" [FAILED] Task {num} exited with code {res.returncode} ({dur:.1f}s)")
        else:
            print(f" [PASSED] Task {num} completed successfully ({dur:.1f}s)")

    total_dur = time.perf_counter() - start_time
    passed = len(TASKS) - len(failures)
    print("\n" + "=" * 65)
    print(f" ALL TASKS SUMMARY: {passed}/{len(TASKS)} Passed in {total_dur:.1f}s")
    print("=" * 65)

    if failures:
        for num, title, code in failures:
            print(f"  - Failed Task {num} ({title}): Code {code}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
