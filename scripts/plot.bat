cd /d "%~dp0.."

:: Check for venv and activate
if exist "venv\Scripts\activate.bat" (
    echo Activating venv...
    call venv\Scripts\activate.bat
) else (
    echo Warning: venv not found. Using system python.
)

echo Running Plotly Visualization...
python -m src.plot_model
