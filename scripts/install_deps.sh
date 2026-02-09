#!/bin/bash
set -e

cd "$(dirname "$0")/.."


echo "Installing system dependencies..."
# sudo apt-get update
# Install python3, pip, and venv if not present. 
# Also build-essential might be needed for some python packages like Prophet (pystan) if no wheel exists, 
# but prophet usually has wheels now.
sudo apt-get install -y python3 python3-pip python3-venv build-essential

echo "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Installation complete."
echo "Creating logs directory..."
mkdir -p logs
echo "Please edit 'settings.toml' with your credentials before running the system."
