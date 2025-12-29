#!/usr/bin/env python3
"""
Development server for Piano Transcription Web Platform

Usage:
    python run_server.py                    # Run on localhost:8000
    python run_server.py --host 0.0.0.0    # Run on all interfaces
    python run_server.py --port 8080       # Custom port
    python run_server.py --reload          # Enable auto-reload for development
"""

import argparse
import uvicorn
from pathlib import Path
import sys

# Add parent directory to path to import piano_transcriber
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    parser = argparse.ArgumentParser(description="Piano Transcription Web Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    print("🎹 Starting Piano Transcription Web Server...")
    print(f"📍 Server will be available at: http://{args.host}:{args.port}")
    print("🎵 Upload audio files to convert piano recordings to MIDI!")
    print("\n" + "="*50)
    
    # Change to web directory for proper static file serving
    import os
    web_dir = Path(__file__).parent
    os.chdir(web_dir)
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )

if __name__ == "__main__":
    main()