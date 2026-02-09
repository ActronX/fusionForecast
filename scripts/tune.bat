@echo off
cd /d "%~dp0.."

REM Activate venv if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating venv...
    call venv\Scripts\activate.bat
) else (
    echo Warning: venv not found. Using system python.
)

echo Running hyperparameter tuning...
echo This may take a while depending on the parameter grid.
echo.
python -m src.tune
if %ERRORLEVEL% NEQ 0 (
    echo Execution failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo Done. Results saved to tuning_results.csv

