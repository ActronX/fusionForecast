#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# Required Python version
PYTHON_VERSION="3.11"
PYTHON_CMD="python${PYTHON_VERSION}"

echo "============================================"
echo "FusionForecast - Dependency Installer"
echo "============================================"

# Check if required Python version is already available
if command -v "$PYTHON_CMD" &> /dev/null; then
    echo "Python $PYTHON_VERSION found: $($PYTHON_CMD --version)"
else
    echo "Python $PYTHON_VERSION not found. Installing via deadsnakes PPA..."
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv" "python${PYTHON_VERSION}-dev"
    echo "Python $PYTHON_VERSION installed: $($PYTHON_CMD --version)"
fi

echo "Installing build tools..."
sudo apt-get install -y build-essential

echo ""
echo "Setting up Python virtual environment..."
if [ -d "venv" ]; then
    # Check if existing venv uses the correct Python version
    VENV_PYTHON_VERSION=$(venv/bin/python3 --version 2>/dev/null | grep -oP '\d+\.\d+' || echo "unknown")
    if [ "$VENV_PYTHON_VERSION" != "$PYTHON_VERSION" ]; then
        echo "Existing venv uses Python $VENV_PYTHON_VERSION, recreating with $PYTHON_VERSION..."
        rm -rf venv
        $PYTHON_CMD -m venv venv
    else
        echo "Existing venv already uses Python $PYTHON_VERSION."
    fi
else
    $PYTHON_CMD -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Creating logs directory..."
mkdir -p logs

echo ""
echo "============================================"
echo "Installation complete."
echo "Python: $(python3 --version)"
echo "============================================"
echo "Please edit 'settings.toml' with your credentials before running the system."
