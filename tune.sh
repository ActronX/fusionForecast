#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Starting Hyperparameter Tuning..."
echo "This process may take a while depending on the grid size in settings.toml."

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: venv not found. Using system python."
fi

python3 src/tune.py
