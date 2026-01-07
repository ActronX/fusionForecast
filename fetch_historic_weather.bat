@echo off
cd /d "%~dp0"

REM Activate venv if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating venv...
    call venv\Scripts\activate.bat
) else (
    echo Warning: venv not found. Using system python.
)

echo Running historic Regressor/Weather data fetcher...
python -m src.fetch_historic_weather
if %ERRORLEVEL% NEQ 0 (
    echo Execution failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo Done.
pause
