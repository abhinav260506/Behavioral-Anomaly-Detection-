@echo off
title Aegis AI - SOC Analyst Dashboard
setlocal enabledelayedexpansion

echo ======================================================================
echo           Aegis AI: SOC Behavioral Anomaly Detection Engine
echo ======================================================================
echo.

:: Verify Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3 was not detected in your system PATH.
    echo Please install Python 3.10+ from python.org and check "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

echo [*] Python environment detected.
echo [*] Verifying dependencies from requirements.txt...
pip install -r requirements.txt --quiet

echo.
echo [*] Starting Aegis AI Dashboard Server...
echo [*] Dashboard will be available at: http://localhost:5000
echo [*] Press Ctrl+C in this window at any time to stop the server.
echo.

:: Automatically open browser
start "" http://localhost:5000

:: Launch application
python app.py

pause
