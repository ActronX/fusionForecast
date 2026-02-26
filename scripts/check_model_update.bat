@echo off
cd /d "%~dp0.."

REM Activate venv if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating venv...
    call venv\Scripts\activate.bat
) else (
    echo Warning: venv not found. Using system python.
)

REM Default model: dwd_icon_d2_15min
set MODEL=%1
if "%MODEL%"=="" set MODEL=dwd_icon_d2_15min

echo Checking for model update (%MODEL%)...
python -m src.check_model_update %MODEL%
set RESULT=%ERRORLEVEL%

if %RESULT% EQU 0 (
    echo Update available.
) else if %RESULT% EQU 1 (
    echo No update.
) else (
    echo Check failed with error code %RESULT%
)

exit /b %RESULT%
