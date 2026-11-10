@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -u comparison_app.py --open-browser
if errorlevel 1 pause
