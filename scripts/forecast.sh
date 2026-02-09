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

echo "Running forecast pipeline..."
# Run as module
python3 -m src.forecast
