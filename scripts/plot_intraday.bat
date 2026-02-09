@echo off
cd /d "%~dp0.."
echo Starting Intraday AR Plotter...
python src/plot_intraday_ar.py
if errorlevel 1 (
    echo Error executing script.
)
