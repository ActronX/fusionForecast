@echo off
cd /d "%~dp0.."
echo Building FusionForecast components...

echo Building FusionForecast Pipeline (run_pipeline.py)...
pyinstaller --onefile --name FusionForecastPipeline --paths . --distpath dist --workpath build --clean --collect-all prophet run_pipeline.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo Copying settings.toml to dist folder...
copy settings.toml dist\settings.toml

echo Cleaning up spec files...
del /Q *.spec

echo Build complete. Executables are in the 'dist' folder.
