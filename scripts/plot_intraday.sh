#!/bin/bash
cd "$(dirname "$0")/.."
echo "Starting Intraday AR Plotter..."
python src/plot_intraday_ar.py
