@echo off
cd /d "%~dp0"
set PYTHONPATH=%PYTHONPATH%;%CD%
python test_connection.py
pause
