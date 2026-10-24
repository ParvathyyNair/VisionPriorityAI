@echo off
REM Run ONCE on a computer WITH internet (same OS + Python version as the target laptop).
REM Downloads every dependency into .\wheelhouse so the app can be installed with no internet.
cd /d "%~dp0"
python -m pip download -r requirements.txt -d wheelhouse
echo.
echo Done. Copy this whole folder (with wheelhouse) to the offline laptop and run install_offline.bat
pause
