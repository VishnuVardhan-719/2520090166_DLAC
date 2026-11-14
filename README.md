# Deep Learning Academic Coursework (DLAC)
**Student ID:** 2520090166  
**Project Title:** VisionPort – Intelligent Vehicle Detection and Traffic Monitoring Using Deep Learning  

---

## 📁 Repository Structure

```text
2520090166_DLAC/
└── VisionPort/                    # Complete Project Implementation
    ├── samples/                   # Input media (intersection, signal, congestion, video)
    ├── tasks/                     # Individual student task programs (Tasks 1 to 15)
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
    ├── templates/                 # Web Portal UI
    │   └── index.html             # Interactive evaluation dashboard
    ├── main.py                    # Local FastAPI evaluation server
    ├── run_all_tasks.py           # Automated batch runner for all 15 tasks
    ├── run_website.bat            # 1-Click launcher for web portal
    ├── requirements.txt           # Python dependencies
    └── yolov8n.pt                 # YOLOv8 pre-trained weights
```

---

## 🚀 Quick Start Guide

### 1. Run Interactive Web Portal
Launch the local web evaluation dashboard:
```powershell
cd VisionPort
python main.py
```
Or simply double-click **`run_website.bat`**.  
Open in your browser: **http://127.0.0.1:8000**

### 2. Run All 15 Tasks in Batch
```powershell
cd VisionPort
python run_all_tasks.py
```

### 3. Run Any Individual Task
```powershell
cd VisionPort
python tasks/task6_detect_cars.py
python tasks/task12_track_vehicles.py
python tasks/task15_estimate_waiting_time.py
```

---

## 📋 Task Mapping (Tasks 1–15)

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

