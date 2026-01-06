@echo off
echo Starting FusionForecast Pipeline...
python run_pipeline.py
if %errorlevel% neq 0 (
    echo Pipeline failed with error code %errorlevel%
) else (
    echo Pipeline finished successfully.
)
