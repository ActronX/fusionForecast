#!/bin/bash
echo "Testing connection..."
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 test_connection.py
read -p "Press Enter to exit..."
