"""Auto-start helpers — detect and launch Vortelio server if needed."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def find_vortelio_exe() -> Optional[str]:
    # PATH lookup
    exe = shutil.which("vortelio") or shutil.which("vortelio.exe")
    if exe:
        return exe
    # Common install locations
    candidates = [
        Path.home() / ".cache" / "vortelio",
        Path.home() / ".local" / "bin",
        Path("/usr/local/bin"),
    ]
    for d in candidates:
        for name in ("vortelio", "vortelio.exe"):
            p = d / name
            if p.exists():
                return str(p)
    return None


def is_server_running(port: int = 11500) -> bool:
    try:
        urllib.request.urlopen(f"http://localhost:{port}/api/version", timeout=2)
        return True
    except Exception:
        return False


def ensure_server(port: int = 11500, wait: float = 8.0) -> bool:
    if is_server_running(port):
        return True
    exe = find_vortelio_exe()
    if not exe:
        return False
    try:
        subprocess.Popen(
            [exe, "serve", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return False
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if is_server_running(port):
            return True
        time.sleep(0.4)
    return False


def install_vortelio() -> bool:
    """Attempt to install vortelio via pip (pulls the CLI binary package)."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "vortelio"],
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


# Re-export Optional for setup.py's own use (avoids import at module level)
from typing import Optional  # noqa: E402
