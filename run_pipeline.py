
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import train_model
from src.forecast import run_forecast

def main():
    print("=== Starting Pipeline ===")
    
    print("\n--- Step 1: Training ---")
    try:
        train_model()
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)
        
    print("\n--- Step 2: Forecasting ---")
    try:
        run_forecast()
    except Exception as e:
        print(f"Error during forecasting: {e}")
        sys.exit(1)
        
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
