@echo off
echo Starting FusionForecast Pipeline...
cd /d "%~dp0"

REM Activate venv if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating venv...
    call venv\Scripts\activate.bat
) else (
    echo Warning: venv not found. Using system python.
)

python run_pipeline.py
if %errorlevel% neq 0 (
    echo Pipeline failed with error code %errorlevel%
) else (
    echo Pipeline finished successfully.
)
