#!/bin/bash
set -e

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: venv not found. Using system python."
fi

echo "Starting FusionForecast Pipeline..."
python3 run_pipeline.py

if [ $? -eq 0 ]; then
    echo "Pipeline finished successfully."
else
    echo "Pipeline failed with error code $?"
fi
