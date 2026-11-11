@echo off
echo.
echo 🚦 Traffic Neural Network System
echo ==================================================
echo.
echo 🌐 Starting application server...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.6+ from: https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM Check if build directory exists
if not exist "build" (
    echo ❌ Error: build directory not found!
    echo Please run 'npm run build' first to create the production build.
    echo.
    pause
    exit /b 1
)

REM Run the Python server
echo ✅ Starting Traffic Neural Network Application...
echo 🔗 The application will open in your browser automatically
echo 🛑 Press Ctrl+C to stop the server
echo.
python run_app.py

echo.
echo 👋 Thank you for using Traffic Neural Network System!
pause
