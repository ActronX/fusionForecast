@echo off
cd /d "%~dp0"
set PYTHONPATH=%PYTHONPATH%;%CD%
python src/train_consumption.py
pause
