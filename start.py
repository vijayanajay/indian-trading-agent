#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import urllib.request
import webbrowser
import signal
import socket


# Configuration
BACKEND_PORT = os.environ.get("BACKEND_PORT", "8000")
FRONTEND_PORT = os.environ.get("FRONTEND_PORT", "3000")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

backend_proc = None
frontend_proc = None

def log(msg):
    print(f"\033[94m[start.py]\033[0m {msg}")

def err(msg):
    print(f"\033[91m[error]\033[0m {msg}", file=sys.stderr)

def ok(msg):
    print(f"\033[92m[ok]\033[0m {msg}")

def kill_process_on_port(port):
    """Find and kill any process using a given port on Windows/Linux/macOS."""
    if sys.platform == "win32":
        try:
            # Get netstat output for listening ports
            output = subprocess.check_output(
                f'netstat -ano | findstr LISTENING | findstr :{port}', 
                shell=True, 
                text=True
            )
            pids = set()
            for line in output.strip().split("\n"):
                parts = line.strip().split()
                if parts:
                    pid = parts[-1]
                    pids.add(pid)
            for pid in pids:
                log(f"Port {port} is in use by PID {pid}. Killing it...")
                subprocess.run(
                    f"taskkill /F /PID {pid}", 
                    shell=True, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
        except subprocess.CalledProcessError:
            pass # Port not in use
    else:
        try:
            pids = subprocess.check_output(
                f"lsof -t -i:{port}", 
                shell=True, 
                text=True
            )
            for pid in pids.strip().split("\n"):
                if pid:
                    log(f"Port {port} is in use by PID {pid}. Killing it...")
                    subprocess.run(f"kill -9 {pid}", shell=True)
        except subprocess.CalledProcessError:
            pass

def cleanup(signum=None, frame=None):
    log("Shutting down processes...")
    if backend_proc:
        log("Stopping backend...")
        try:
            backend_proc.terminate()
            backend_proc.wait(timeout=3)
        except Exception:
            try:
                backend_proc.kill()
            except Exception:
                pass
    if frontend_proc:
        log("Stopping frontend...")
        try:
            frontend_proc.terminate()
            frontend_proc.wait(timeout=3)
        except Exception:
            try:
                frontend_proc.kill()
            except Exception:
                pass
    log("Stopped.")
    sys.exit(0)

# Register signals for clean exit
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def check_port_listening(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a TCP port is listening on the specified host."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

def wait_for_backend(backend_url, proc, timeout_seconds=30):
    """Wait for backend HTTP endpoint to return 200 OK."""
    log("Waiting for backend to be ready...")
    for _ in range(timeout_seconds):
        if proc and hasattr(proc, 'poll') and proc.poll() is not None:
            err("Backend process died unexpectedly.")
            return False
        try:
            with urllib.request.urlopen(backend_url, timeout=1) as response:
                if response.status == 200:
                    ok(f"Backend ready at {backend_url}")
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False

def wait_for_frontend(host, port, proc, timeout_seconds=30):
    """Wait for frontend TCP port to be listening."""
    log("Waiting for frontend to be ready...")
    for _ in range(timeout_seconds):
        if proc and hasattr(proc, 'poll') and proc.poll() is not None:
            err("Frontend process died unexpectedly.")
            return False
        if check_port_listening(host, port):
            ok(f"Frontend ready at http://{host}:{port}")
            return True
        time.sleep(1)
    return False

def main():
    global backend_proc, frontend_proc
    os.chdir(ROOT_DIR)

    # 1. Pre-flight checks
    if not os.path.exists("frontend/node_modules"):
        log("frontend/node_modules missing. Running 'npm install' (one-time setup)...")
        subprocess.run("npm install", shell=True, cwd=os.path.join(ROOT_DIR, "frontend"))

    # 2. Free ports
    kill_process_on_port(BACKEND_PORT)
    kill_process_on_port(FRONTEND_PORT)

    # Determine Python command
    python_cmd = sys.executable

    # 3. Start backend
    log(f"Starting backend on port {BACKEND_PORT}...")
    backend_env = os.environ.copy()
    backend_proc = subprocess.Popen(
        [python_cmd, "-m", "uvicorn", "backend.app:app", "--reload", "--port", str(BACKEND_PORT)],
        cwd=ROOT_DIR,
        env=backend_env
    )

    # 4. Start frontend
    log(f"Starting frontend on port {FRONTEND_PORT}...")
    frontend_proc = subprocess.Popen(
        "npm run dev -- --port " + str(FRONTEND_PORT),
        shell=True,
        cwd=os.path.join(ROOT_DIR, "frontend")
    )

    # 5. Health checks
    backend_url = f"http://localhost:{BACKEND_PORT}/api/health"
    frontend_url = f"http://localhost:{FRONTEND_PORT}"

    if not wait_for_backend(backend_url, backend_proc):
        err("Backend health check timed out.")
        cleanup()

    if not wait_for_frontend("localhost", int(FRONTEND_PORT), frontend_proc):
        err("Frontend health check timed out.")
        cleanup()

    # 6. Open browser
    log(f"Opening {frontend_url} in browser...")
    webbrowser.open(frontend_url)

    ok("Both servers are running. Press Ctrl+C to stop everything.")
    
    # Keep main thread alive to manage subprocesses
    try:
        while True:
            if backend_proc.poll() is not None:
                err("Backend process stopped.")
                break
            if frontend_proc.poll() is not None:
                err("Frontend process stopped.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()

if __name__ == "__main__":
    main()
