#!/usr/bin/env sh
# VisionPriority AI - start the dashboard locally. Works with NO internet connection.
cd "$(dirname "$0")"
export YOLO_OFFLINE=true YOLO_AUTOINSTALL=false
python3 scripts/offline_check.py --quick
exec python3 -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
