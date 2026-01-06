@echo off
cd /d "%~dp0"
set PYTHONPATH=%PYTHONPATH%;%CD%
python src/forecast_consumption.py
pause
