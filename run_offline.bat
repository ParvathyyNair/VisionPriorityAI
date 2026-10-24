@echo off
REM VisionPriority AI - start the dashboard locally. Works with NO internet connection.
cd /d "%~dp0"
set YOLO_OFFLINE=true
set YOLO_AUTOINSTALL=false
python scripts\offline_check.py --quick
python -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
pause
