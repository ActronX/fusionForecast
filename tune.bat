@echo off
cd /d "%~dp0"
echo Starting Hyperparameter Tuning...
echo This process may take a while depending on the grid size in settings.toml.

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Warning: venv not found. Using system python.
)

python src/tune.py
pause
