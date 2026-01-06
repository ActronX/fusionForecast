#!/bin/bash
cd "$(dirname "$0")"

# Activate venv if it exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating venv..."
    source venv/bin/activate
else
    echo "Warning: venv not found. Using system python."
fi

echo "Running historic DWD data fetcher..."
python3 -m src.fetch_historic_dwd_data
if [ $? -ne 0 ]; then
    echo "Execution failed with error code $?"
    exit $?
fi

echo "Done."
