#!/bin/bash

echo ""
echo "🚦 Traffic Neural Network System"
echo "=================================================="
echo ""
echo "🌐 Starting application server..."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH!"
    echo ""
    echo "Please install Python 3.6+ from your package manager:"
    echo "  Ubuntu/Debian: sudo apt install python3"
    echo "  macOS: brew install python3"
    echo "  Or download from: https://python.org"
    echo ""
    exit 1
fi

# Check if build directory exists
if [ ! -d "build" ]; then
    echo "❌ Error: build directory not found!"
    echo "Please run 'npm run build' first to create the production build."
    echo ""
    exit 1
fi

# Make the Python script executable
chmod +x run_app.py

# Run the Python server
echo "✅ Starting Traffic Neural Network Application..."
echo "🔗 The application will open in your browser automatically"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

python3 run_app.py

echo ""
echo "👋 Thank you for using Traffic Neural Network System!"
