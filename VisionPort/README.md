# VisionPort – Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning

VisionPort is an end-to-end intelligent traffic monitoring system built with Python, OpenCV 5.0, and Ultralytics YOLOv8. It covers all 15 student project tasks individually and provides a modern interactive evaluation web portal.

---

## 📁 Clean Project Structure

```text
VisionPort/
├── .venv/                         # Virtual environment (Python 3.14 + OpenCV + PyTorch + YOLOv8)
├── samples/                       # Test inputs (images and video)
│   ├── traffic_congested.jpg      # Congested traffic test image
│   ├── traffic_intersection.jpg   # Multi-vehicle urban junction test image
│   ├── traffic_sample.mp4         # Multi-phase traffic video (Red -> Green -> Yellow)
│   └── traffic_signal.jpg         # Traffic signal test image
├── tasks/                         # Dedicated scripts for Tasks 1 to 15
│   ├── task1_setup_and_verify.py
│   ├── task2_capture_traffic_signal.py
│   ├── task3_read_display_traffic_image.py
│   ├── task4_load_play_traffic_video.py
│   ├── task5_extract_video_frames.py
│   ├── task6_detect_cars.py
│   ├── task7_detect_multiple_vehicles.py
│   ├── task8_display_labels_confidence.py
│   ├── task9_count_vehicles.py
│   ├── task10_traffic_density.py
│   ├── task11_detect_congestion.py
│   ├── task12_track_vehicles.py
│   ├── task13_calculate_vehicle_speed.py
│   ├── task14_identify_traffic_signal.py
│   └── task15_estimate_waiting_time.py
├── templates/                     # Web evaluation portal template
│   └── index.html                 # Interactive UI for testing all tasks
├── .gitignore                     # Git ignore rules
├── main.py                        # FastAPI web application server
├── README.md                      # Project documentation
├── requirements.txt               # Pinned dependencies
├── run_all_tasks.py               # Automated batch verification for all 15 tasks
└── yolov8n.pt                     # YOLOv8 Nano pre-trained weights
```

---

## 🚀 Running the Tasks

### 1. Interactive Web Portal
Launch the local web server to select any task, upload images/videos, use live camera capture, and observe real-time telemetry:

```powershell
.\.venv\Scripts\python.exe main.py
```
Open **http://127.0.0.1:8000** in your web browser.

### 2. Run All 15 Tasks in Batch
To verify that all 15 task programs run and pass end-to-end:

```powershell
.\.venv\Scripts\python.exe run_all_tasks.py
```

### 3. Run Individual Tasks Standalone
Run any task directly with its own script:

```powershell
# Task 1: Environment Verification
.\.venv\Scripts\python.exe tasks\task1_setup_and_verify.py

# Task 2: Camera Capture
.\.venv\Scripts\python.exe tasks\task2_capture_traffic_signal.py

# Task 3: Read & Display Image
.\.venv\Scripts\python.exe tasks\task3_read_display_traffic_image.py

# Task 4: Load & Play Video
.\.venv\Scripts\python.exe tasks\task4_load_play_traffic_video.py

# Task 5: Extract Frames
.\.venv\Scripts\python.exe tasks\task5_extract_video_frames.py

# Task 6: Detect Cars Using YOLO
.\.venv\Scripts\python.exe tasks\task6_detect_cars.py

# Task 7: Detect Bikes, Buses & Trucks
.\.venv\Scripts\python.exe tasks\task7_detect_multiple_vehicles.py

# Task 8: Display Labels & Confidence Scores
.\.venv\Scripts\python.exe tasks\task8_display_labels_confidence.py

# Task 9: Count Vehicles
.\.venv\Scripts\python.exe tasks\task9_count_vehicles.py

# Task 10: Calculate Traffic Density
.\.venv\Scripts\python.exe tasks\task10_traffic_density.py

# Task 11: Detect Congestion Alerts
.\.venv\Scripts\python.exe tasks\task11_detect_congestion.py

# Task 12: Track Vehicles Across Frames
.\.venv\Scripts\python.exe tasks\task12_track_vehicles.py

# Task 13: Calculate Vehicle Speed
.\.venv\Scripts\python.exe tasks\task13_calculate_vehicle_speed.py

# Task 14: Identify Traffic Signal Status
.\.venv\Scripts\python.exe tasks\task14_identify_traffic_signal.py

# Task 15: Estimate Traffic Waiting Time
.\.venv\Scripts\python.exe tasks\task15_estimate_waiting_time.py
```

---

## 📋 Task Matrix

| Task No. | Task Description | Expected Output | Script File |
| :---: | :--- | :--- | :--- |
| **1** | Install Python, OpenCV and required libraries | Working Python environment | `tasks/task1_setup_and_verify.py` |
| **2** | Capture an image of a traffic signal using a camera | Traffic image capture | `tasks/task2_capture_traffic_signal.py` |
| **3** | Read and display traffic images using OpenCV | Image display window | `tasks/task3_read_display_traffic_image.py` |
| **4** | Load and play a traffic video using OpenCV | Video playback | `tasks/task4_load_play_traffic_video.py` |
| **5** | Extract individual frames from traffic videos | Frames saved as images | `tasks/task5_extract_video_frames.py` |
| **6** | Detect cars in a traffic video using YOLO | Car detection with bounding boxes | `tasks/task6_detect_cars.py` |
| **7** | Detect bikes, buses and trucks | Multiple vehicle detection | `tasks/task7_detect_multiple_vehicles.py` |
| **8** | Display vehicle names and confidence scores | Labelled detection output | `tasks/task8_display_labels_confidence.py` |
| **9** | Count the number of vehicles in a video frame | Vehicle count | `tasks/task9_count_vehicles.py` |
| **10** | Calculate traffic density based on vehicle count | Low, medium and high traffic | `tasks/task10_traffic_density.py` |
| **11** | Detect traffic congestion at signal junctions | Congestion alerts | `tasks/task11_detect_congestion.py` |
| **12** | Track individual vehicles across video frames | Vehicle tracking IDs | `tasks/task12_track_vehicles.py` |
| **13** | Calculate approximate vehicle speed | Speed estimation | `tasks/task13_calculate_vehicle_speed.py` |
| **14** | Identify traffic signal status (red, yellow, green) | Signal status detection | `tasks/task14_identify_traffic_signal.py` |
| **15** | Estimate traffic waiting time | Waiting-time report | `tasks/task15_estimate_waiting_time.py` |


