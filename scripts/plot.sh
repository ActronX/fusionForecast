#!/bin/bash
echo "Running Plotly Visualization..."
cd "$(dirname "$0")/.."
python3 -m src.plot_model
read -p "Press Enter to exit..."
