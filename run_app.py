"""
Mint Leaf AI — Prototype Web Application Launcher
Starts the FastAPI Web Server on http://localhost:8000

Usage:
    python run_app.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("=======================================================")
    print("🌿 LAUNCHING MINT LEAF AI REAL-TIME PROTOTYPE SERVER")
    print("=======================================================\n")
    print("  Local URL   : http://localhost:8000")
    print("  API Docs    : http://localhost:8000/docs")
    print("  Press Ctrl+C to stop the server.\n")

    try:
        import uvicorn
    except ImportError:
        print("Installing uvicorn and dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "uvicorn[standard]", "fastapi", "python-multipart",
                        "onnxruntime"], check=True)
        import uvicorn

    # reload=False avoids the Windows multiprocessing bootstrap crash.
    # For development hot-reload use: uvicorn backend.app:app --reload
    uvicorn.run(
        "backend.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )

# Required on Windows: guard the entry point so multiprocessing
# does not re-execute this module in child processes.
if __name__ == "__main__":
    main()
