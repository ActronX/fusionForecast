#!/bin/bash
set -e

# Ensure we are in the script's directory
cd "$(dirname "$0")/.."

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: venv not found. Using system python."
fi

echo "Plotting forecast data..."
# Run as module to handle imports correctly
python3 -m src.plot_forecast_data
