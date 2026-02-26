#!/bin/bash
cd "$(dirname "$0")/.."

# Activate venv if it exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating venv..."
    source venv/bin/activate
else
    echo "Warning: venv not found. Using system python."
fi

# Default model: dwd_icon_d2_15min
MODEL="${1:-dwd_icon_d2_15min}"

echo "Checking for model update ($MODEL)..."
python3 -m src.check_model_update "$MODEL"
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "Update available."
elif [ $RESULT -eq 1 ]; then
    echo "No update."
else
    echo "Check failed with error code $RESULT"
fi

exit $RESULT
