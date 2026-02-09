@echo off
cd /d "%~dp0.."

REM Activate venv if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating venv...
    call venv\Scripts\activate.bat
) else (
    echo Warning: venv not found. Using system python.
)

echo Running forecast pipeline...
python -m src.forecast
if %ERRORLEVEL% NEQ 0 (
    echo Execution failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo Done.

