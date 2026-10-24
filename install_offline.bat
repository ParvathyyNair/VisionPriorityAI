@echo off
REM Installs all dependencies from .\wheelhouse - no internet needed.
cd /d "%~dp0"
python -m pip install --no-index --find-links wheelhouse -r requirements.txt
python scripts\offline_check.py
pause
