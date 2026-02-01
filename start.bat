@echo off
REM ============================================================================
REM Hawkins Truth Engine - Launch Script
REM ============================================================================
REM This script starts the Hawkins Truth Engine server with all dependencies
REM ============================================================================

setlocal EnableDelayedExpansion

REM Set title
title Hawkins Truth Engine

REM Get the directory where this script is located
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo.
echo ============================================================================
echo   HAWKINS TRUTH ENGINE - Credibility Analysis System
echo ============================================================================
echo.

REM Check if Python is available
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Creating...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if dependencies are installed
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    pip install -e . --quiet
    pip install python-dotenv --quiet
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
)

REM Check for .env file
if exist ".env" (
    echo [OK] Environment file found
) else (
    echo [WARNING] No .env file found - using default configuration
    echo          Create a .env file with API keys for full functionality
)

echo.
echo ============================================================================
echo   Starting Server...
echo ============================================================================
echo.
echo   Server URL:    http://127.0.0.1:8000
echo   API Docs:      http://127.0.0.1:8000/docs
echo   Health Check:  http://127.0.0.1:8000/health
echo.
echo   Press Ctrl+C to stop the server
echo ============================================================================
echo.

REM Start the server
python -m hawkins_truth_engine.app --host 127.0.0.1 --port 8000

REM If server exits, pause to show any errors
echo.
echo [INFO] Server stopped
pause
