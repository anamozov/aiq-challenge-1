#!/usr/bin/env python3
"""
Simple HTTP server to serve the frontend HTML file
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

def serve_frontend(port=8080):
    """Serve the frontend HTML file on the specified port"""
    
    # Change to the directory containing the frontend.html file
    frontend_dir = Path(__file__).parent
    os.chdir(frontend_dir)
    
    # Create a simple HTTP server
    handler = http.server.SimpleHTTPRequestHandler
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"🌐 Frontend server starting on port {port}")
        print(f"📁 Serving files from: {frontend_dir}")
        print(f"🔗 Access the frontend at: http://localhost:{port}/frontend.html")
        print(f"🔗 Or from remote: http://10.227.228.64:{port}/frontend.html")
        print("Press Ctrl+C to stop the server")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")

if __name__ == "__main__":
    import sys
    
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port number. Using default port 8080")
    
    serve_frontend(port)
