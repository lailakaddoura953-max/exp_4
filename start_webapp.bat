@echo off
REM =========================================================================
REM  Yard Hazard Inference Dashboard — Quick Launcher
REM =========================================================================
REM  Double-click this file (or run from cmd) to start the dashboard.
REM
REM  What it does:
REM    1. Activates the Python virtual environment (.venv)
REM    2. Starts the Flask backend (port 5000) in a new terminal window
REM    3. Waits a few seconds for the server to initialize
REM    4. Opens the dashboard in your default web browser
REM
REM  To stop: close the "Hazard Dashboard" terminal window (or Ctrl+C in it).
REM =========================================================================

set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

echo.
echo  ====================================================
echo   Yard Hazard Inference Dashboard
echo  ====================================================
echo.
echo  Starting Flask backend on http://localhost:5000 ...
echo.

REM Start the Flask backend in a new terminal window
REM To use 10-minute demo cycle instead of hourly, uncomment the next line:
REM set DASHBOARD_CYCLE_MINUTES=10
start "Hazard Dashboard" cmd /k "cd /d "%PROJECT_DIR%" && .venv\Scripts\activate && set PYTHONPATH=.;src && python -m dashboard.app"

REM Wait for server to start before opening browser
echo  Waiting for server to initialize...
timeout /t 4 /nobreak >nul

REM Open the dashboard in the default browser
echo  Opening browser...
start "" "http://localhost:5000"

echo.
echo  Dashboard is running at: http://localhost:5000
echo  Close the "Hazard Dashboard" terminal window to stop.
echo.
pause
