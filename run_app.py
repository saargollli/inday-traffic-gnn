#!/usr/bin/env python3
"""
Traffic GNN Application Server
A simple HTTP server to run the Traffic Neural Network application.

Usage:
    python3 run_app.py
    
Then open your browser to: http://localhost:8080
"""

import os
import sys
import http.server
import socketserver
import webbrowser
import threading
import time
from pathlib import Path

# Configuration
PORT = int(os.environ.get('PORT', 8080))
BUILD_DIR = "build"

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler to serve React app properly"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BUILD_DIR, **kwargs)
    
    def end_headers(self):
        # Enable CORS for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        # Serve index.html for any route (SPA routing)
        if not os.path.exists(os.path.join(BUILD_DIR, self.path.lstrip('/'))):
            if not self.path.startswith('/static/'):
                self.path = '/index.html'
        return super().do_GET()

def open_browser():
    """Open the browser after a short delay"""
    time.sleep(2)
    webbrowser.open(f'http://localhost:{PORT}')

def main():
    # Check if build directory exists
    if not os.path.exists(BUILD_DIR):
        print(f"❌ Error: {BUILD_DIR} directory not found!")
        print("Please run 'npm run build' first to create the production build.")
        sys.exit(1)
    
    # Check if index.html exists
    if not os.path.exists(os.path.join(BUILD_DIR, 'index.html')):
        print(f"❌ Error: index.html not found in {BUILD_DIR} directory!")
        print("The build seems incomplete. Please run 'npm run build' again.")
        sys.exit(1)
    
    print("🚦 Traffic Neural Network System")
    print("=" * 50)
    print(f"🌐 Starting server on port {PORT}...")
    print(f"📁 Serving files from: {os.path.abspath(BUILD_DIR)}")
    
    try:
        # Create server
        with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
            print(f"✅ Server started successfully!")
            print(f"🔗 Application URL: http://localhost:{PORT}")
            print(f"🔗 Local Network: http://{get_local_ip()}:{PORT}")
            print()
            print("📝 Instructions:")
            print("1. The browser should open automatically")
            print("2. If not, manually open: http://localhost:{PORT}")
            print("3. Press Ctrl+C to stop the server")
            print()
            print("🎮 How to use the application:")
            print("1. Click 'Pre-Train Models' to initialize AI")
            print("2. Click 'Start Simulation' to begin traffic simulation")
            print("3. Click 'Start Learning' for real-time AI learning")
            print("4. Add/remove vehicles and watch AI predictions!")
            print("=" * 50)
            
            # Open browser in a separate thread
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user.")
        print("👋 Thank you for using Traffic Neural Network System!")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Error: Port {PORT} is already in use!")
            print(f"Please either:")
            print(f"1. Stop the process using port {PORT}")
            print(f"2. Change the PORT variable in this script")
        else:
            print(f"❌ Error starting server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

def get_local_ip():
    """Get local IP address for network access"""
    import socket
    try:
        # Connect to a remote address to get local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except:
        return "localhost"

if __name__ == "__main__":
    main()
