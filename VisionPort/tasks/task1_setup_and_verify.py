"""Task 1: Install Python, OpenCV and required libraries.

Task No: 1
Description: Install Python, OpenCV and required libraries.
Expected Output: Working Python environment
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"

REQUIRED_LIBRARIES = [
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


def main():
    print("=" * 65)
    print(" VisionPort – Task 1: Setup & Environment Verification")
    print("=" * 65)
    print(f"Python Version : {platform.python_version()} ({platform.architecture()[0]})")
    print(f"Executable     : {sys.executable}")
    print(f"Platform       : {platform.system()} {platform.release()} ({platform.version()})")
    print("-" * 65)
    print("Verifying Libraries:")

    all_passed = True
    report_lines = [
        "VisionPort Task 1 - Environment Verification Report",
        f"Python: {platform.python_version()} | Executable: {sys.executable}",
        f"Platform: {platform.system()} {platform.release()}",
        "-" * 55,
    ]

    for mod_name, label, _ in REQUIRED_LIBRARIES:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "Installed")
            print(f"  [OK]   {label:<30} -> {ver}")
            report_lines.append(f"{label:<30} : {ver} [OK]")
        except Exception as exc:
            all_passed = False
            print(f"  [FAIL] {label:<30} -> ERROR: {exc}")
            report_lines.append(f"{label:<30} : FAILED ({exc})")

    # PyTorch Accelerator Check
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
        print(f"  [INFO] {'Hardware Accelerator':<30} -> {device_name} (CUDA={cuda_avail})")
        report_lines.append(f"Accelerator: {device_name} (CUDA={cuda_avail})")
    except Exception:
        pass

    # YOLO Weights Check
    if DEFAULT_WEIGHTS.exists():
        size_mb = round(DEFAULT_WEIGHTS.stat().st_size / (1024 * 1024), 2)
        print(f"  [OK]   {'YOLOv8 Nano Weights':<30} -> {DEFAULT_WEIGHTS.name} ({size_mb} MB)")
        report_lines.append(f"YOLOv8 Weights: {DEFAULT_WEIGHTS.name} ({size_mb} MB) [OK]")
    else:
        all_passed = False
        print(f"  [FAIL] {'YOLOv8 Nano Weights':<30} -> MISSING ({DEFAULT_WEIGHTS})")
        report_lines.append("YOLOv8 Weights: MISSING [FAIL]")

    print("-" * 65)
    overall_status = "Working Python environment" if all_passed else "Incomplete Python environment"
    print(f"Expected Output: {overall_status}")
    print("=" * 65)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

