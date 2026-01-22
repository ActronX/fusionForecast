#!/bin/bash
cd "$(dirname "$0")"

# Activate venv if it exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating venv..."
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    echo "Activating venv..."
    source venv/Scripts/activate
else
    echo "Warning: venv not found. Using system python."
fi

echo "Testing connection..."
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 test_connection.py
read -p "Press Enter to exit..."
