@echo off
title VisionPort – Intelligent Traffic Monitoring Dashboard
cd /d "%~dp0"
echo =================================================================
echo  Starting VisionPort AI Dashboard on http://127.0.0.1:8000 ...
echo =================================================================
echo.
echo Opening browser...
start http://127.0.0.1:8000
echo.
echo Launching FastAPI Server with YOLOv8...
.\.venv\Scripts\python.exe main.py
pause

