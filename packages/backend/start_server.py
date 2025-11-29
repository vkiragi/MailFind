#!/usr/bin/env python3
"""
Smart server startup script that finds an available port.
If port 8000 is in use, it will try subsequent ports (8001, 8002, etc.)
"""

import socket
import subprocess
import sys
import os


def is_port_available(port: int) -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False


def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts - 1}")


def main():
    try:
        # Find an available port
        port = find_available_port()

        if port != 8000:
            print(f"⚠️  Port 8000 is in use. Starting server on port {port} instead.")
            print(f"📝 Update your Chrome extension to use: http://localhost:{port}")
        else:
            print(f"✓ Starting server on port {port}")

        # Start uvicorn with the available port
        cmd = [
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--reload",
            "--port", str(port),
            "--host", "0.0.0.0"
        ]

        print(f"\n🚀 Running: {' '.join(cmd)}\n")

        # Run uvicorn and forward all output
        subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
