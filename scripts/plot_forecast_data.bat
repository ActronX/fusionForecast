@echo off
cd /d "%~dp0.."

REM Activate venv if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating venv...
    call venv\Scripts\activate.bat
) else (
    echo Warning: venv not found. Using system python.
)

REM Usage:
REM   plot_forecast_data.bat              -> forecast results (yhat)
REM   plot_forecast_data.bat regressors   -> regressor inputs (forecast_data.csv)

set MODE=forecast
if /I "%~1"=="regressors" set MODE=regressors

echo Plotting %MODE% data...
python -m src.plot_forecast_data --mode %MODE%
if %ERRORLEVEL% NEQ 0 (
    echo Execution failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo Done.
