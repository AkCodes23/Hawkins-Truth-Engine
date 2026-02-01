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

REM Check Python version
echo [INFO] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to get Python version
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python version: %PYTHON_VERSION%

REM Verify Python version is 3.11 or higher
python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.11 or higher is required
    echo         Current version: %PYTHON_VERSION%
    echo         Please install Python 3.11+ from https://python.org
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
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo [INFO] Ensuring pip is up to date...
python -m pip install --upgrade pip --quiet >nul 2>&1

REM Check if dependencies are installed
echo [INFO] Checking dependencies...
python -c "import fastapi" >nul 2>&1
if errorlevel 1 goto :install_deps
python -c "import uvicorn" >nul 2>&1
if errorlevel 1 goto :install_deps
python -c "import pydantic" >nul 2>&1
if errorlevel 1 goto :install_deps
python -c "import httpx" >nul 2>&1
if errorlevel 1 goto :install_deps
python -c "import trafilatura" >nul 2>&1
if errorlevel 1 goto :install_deps
python -c "import numpy" >nul 2>&1
if errorlevel 1 goto :install_deps
python -c "import sklearn" >nul 2>&1
if errorlevel 1 goto :install_deps
python -c "import rapidfuzz" >nul 2>&1
if errorlevel 1 goto :install_deps
echo [OK] Core dependencies already installed
goto :deps_done

:install_deps
echo [INFO] Installing/updating dependencies - this may take a few minutes...
echo [INFO] Installing package in editable mode to use latest code...
pip install -e . --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    echo [INFO] Trying without quiet mode to see errors...
    pip install -e .
    if errorlevel 1 (
        echo [ERROR] Installation failed. Please check the errors above.
        pause
        exit /b 1
    )
)
echo [OK] Dependencies installed/updated

:deps_done

REM Check for python-dotenv
python -c "import dotenv" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing python-dotenv...
    pip install python-dotenv --quiet
    if errorlevel 1 (
        echo [WARNING] Failed to install python-dotenv - .env file loading may not work
    ) else (
        echo [OK] python-dotenv installed
    )
)

REM Check for .env file
if exist ".env" (
    echo [OK] Environment file .env found
) else (
    echo [WARNING] No .env file found - using default configuration
    echo          Optional: Create a .env file with API keys for full functionality
    echo          - HTE_NCBI_EMAIL=your.email@example.com
    echo          - HTE_NCBI_API_KEY=your_api_key - optional
    echo          - HTE_TAVILY_API_KEY=your_api_key - optional
    echo          - HTE_GROQ_API_KEY=your_api_key - optional
)

REM Verify the package is installed correctly and ensure latest code is used
echo [INFO] Verifying installation and ensuring latest code...
python -c "import hawkins_truth_engine" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Package not installed - installing in editable mode...
    pip install -e . --force-reinstall --no-deps
    pip install -e .
    if errorlevel 1 (
        echo [ERROR] Failed to install package
        pause
        exit /b 1
    )
) else (
    REM Reinstall in editable mode to ensure latest code changes are reflected
    echo [INFO] Reinstalling package in editable mode to use latest code...
    pip install -e . --force-reinstall --no-deps >nul 2>&1
    pip install -e . --quiet >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] Package reinstall had issues - continuing anyway
    ) else (
        echo [OK] Package updated to latest code
    )
)

REM Display version information
echo [INFO] Package version check...
python -c "import hawkins_truth_engine; print('[OK] Package version:', hawkins_truth_engine.__version__)" 2>nul
if errorlevel 1 (
    echo [WARNING] Could not determine package version
)

REM Check if port 8000 is available
echo [INFO] Checking if port 8000 is available...
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Port 8000 is already in use
    echo [INFO] Attempting to find and close existing server...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
        if not "%%a"=="" (
            echo [INFO] Found process using port 8000: PID %%a
            echo [INFO] Closing process...
            taskkill /PID %%a /F >nul 2>&1
            if errorlevel 1 (
                echo [WARNING] Could not close process %%a - you may need to close it manually
            ) else (
                echo [OK] Process closed
            )
        )
    )
    echo [INFO] Waiting 2 seconds for port to be released...
    timeout /t 2 /nobreak >nul
    REM Check again after cleanup
    netstat -ano | findstr ":8000" | findstr "LISTENING" >nul 2>&1
    if not errorlevel 1 (
        echo [ERROR] Port 8000 is still in use
        echo [INFO] Please close the application using port 8000 manually
        echo        You can find it with: netstat -ano ^| findstr ":8000"
        pause
        exit /b 1
    )
    echo [OK] Port 8000 is now available
) else (
    echo [OK] Port 8000 is available
)

echo.
echo ============================================================================
echo   Starting Server...
echo ============================================================================
echo.
echo   Server URL:    http://127.0.0.1:8000
echo   Web Interface:  http://127.0.0.1:8000/
echo   API Docs:      http://127.0.0.1:8000/docs
echo   Health Check:  http://127.0.0.1:8000/health
echo   Status:        http://127.0.0.1:8000/status
echo.
echo   Features Available:
echo   - Content Analysis (text, URL, social posts)
echo   - Multi-signal Analysis (linguistic, statistical, source, claims)
echo   - Graph Visualization (claim graphs, evidence graphs)
echo   - Confidence Calibration
echo   - External API Integration (GDELT, PubMed, RDAP, Tavily)
echo.
echo   Press Ctrl+C to stop the server
echo ============================================================================
echo.

REM Start the server
python -m hawkins_truth_engine.app --host 127.0.0.1 --port 8000

REM If server exits, pause to show any errors
echo.
echo ============================================================================
echo [INFO] Server stopped
echo ============================================================================
pause
