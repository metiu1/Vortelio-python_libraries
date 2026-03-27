"""
Vortelio auto-installer.
Checks if Vortelio is installed, installs it if not, then starts the server.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

GITHUB_RELEASES_API = "https://api.github.com/repos/vortelio/vortelio/releases/latest"
DOWNLOAD_PAGE       = "https://github.com/vortelio/vortelio/releases/latest"


# ── Locate vortelio.exe ────────────────────────────────────────────────────────

def find_vortelio_exe() -> str | None:
    """Return path to vortelio.exe / vortelio if installed, else None."""
    exe = "vortelio.exe" if platform.system() == "Windows" else "vortelio"

    # 1. Standard PATH lookup
    found = shutil.which(exe) or shutil.which("vortelio")
    if found:
        return found

    # 2. On Windows: read PATH from registry (works even if terminal session
    #    predates the installation and hasn't picked up the new PATH yet)
    if platform.system() == "Windows":
        try:
            import winreg
            paths: list[str] = []
            for hive, key in [
                (winreg.HKEY_LOCAL_MACHINE,
                 r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                (winreg.HKEY_CURRENT_USER, "Environment"),
            ]:
                try:
                    with winreg.OpenKey(hive, key) as k:
                        val, _ = winreg.QueryValueEx(k, "Path")
                        paths.append(val)
                except FileNotFoundError:
                    pass
            for folder in ";".join(paths).split(";"):
                candidate = Path(folder.strip()) / exe
                if candidate.exists():
                    return str(candidate)
        except Exception:
            pass

        # 3. Common install folders
        for folder in [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Vortelio",
            Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Vortelio",
            Path(r"C:\Program Files\Vortelio"),
            Path(r"C:\Vortelio"),
            Path.home() / ".vortelio" / "bin",
        ]:
            candidate = folder / exe
            if candidate.exists():
                return str(candidate)

    return None


# ── Check server ─────────────────────────────────────────────────────────────

def is_server_running(port: int = 11500) -> bool:
    """Return True if the Vortelio server is already listening."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/status", timeout=2
        ) as r:
            return r.status == 200
    except Exception:
        return False


# ── Start server ─────────────────────────────────────────────────────────────

def start_server(exe: str, port: int = 11500) -> bool:
    """Launch vortelio serve and wait up to 30 s for it to respond."""
    print(f"🚀  Starting Vortelio server...")
    try:
        if platform.system() == "Windows":
            subprocess.Popen(
                [exe, "serve", "--port", str(port)],
                creationflags=subprocess.DETACHED_PROCESS |
                              subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                [exe, "serve", "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
    except Exception as e:
        print(f"⚠️   Could not launch {exe}: {e}")
        print(f"    Run manually: vortelio serve")
        return False

    print("   Waiting for server ", end="", flush=True)
    for i in range(60):          # 30 seconds
        time.sleep(0.5)
        if is_server_running(port):
            print(" ✅  Ready!")
            return True
        if i % 4 == 3:
            print(".", end="", flush=True)

    print()
    print("⚠️   Server did not respond within 30 seconds.")
    print(f"    Open a new terminal and run:  vortelio serve")
    print(f"    Then re-run your script.")
    return False


# ── Download & install Vortelio ─────────────────────────────────────────────────

def _get_latest_release() -> dict | None:
    try:
        req = urllib.request.Request(
            GITHUB_RELEASES_API,
            headers={"User-Agent": "vortelio-python-sdk"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
            return data if data.get("assets") else None
    except Exception:
        return None


def _pick_installer(assets: list) -> dict | None:
    """Pick the Windows installer asset."""
    system  = platform.system().lower()
    machine = platform.machine().lower()
    arch    = "amd64" if machine in ("x86_64", "amd64") else machine

    for asset in assets:
        name = asset["name"].lower()
        if system == "windows" and "setup" in name and arch in name:
            return asset
    # Fallback: any windows installer
    for asset in assets:
        name = asset["name"].lower()
        if system == "windows" and "setup" in name and ".exe" in name:
            return asset
    # Fallback: portable exe
    for asset in assets:
        name = asset["name"].lower()
        if system == "windows" and "windows" in name and name.endswith(".exe"):
            return asset
    return None


def _download(url: str, dest: Path) -> None:
    """Download url to dest with a progress bar."""
    req = urllib.request.Request(url, headers={"User-Agent": "vortelio-python-sdk"})
    with urllib.request.urlopen(req, timeout=600) as r:
        total      = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                    print(
                        f"\r  [{bar}] {pct:3d}%  "
                        f"{downloaded/1e6:.1f}/{total/1e6:.1f} MB",
                        end="", flush=True,
                    )
    print()


def install_vortelio() -> str | None:
    """
    Download Vortelio-Setup-x.x.x.exe from GitHub and run the silent installer.
    Returns path to vortelio.exe after installation, or None on failure.
    """
    if platform.system() != "Windows":
        print()
        print("  Automatic installation is currently Windows-only.")
        print(f"  Download for your platform from: {DOWNLOAD_PAGE}")
        return None

    print("🔍  Fetching latest Vortelio release from GitHub...")
    release = _get_latest_release()

    if not release:
        print("❌  Could not reach GitHub releases.")
        print(f"    Download manually from: {DOWNLOAD_PAGE}")
        print(f"    Then run: vortelio serve")
        return None

    version = release.get("tag_name", "?")
    asset   = _pick_installer(release.get("assets", []))

    if not asset:
        print(f"❌  No Windows installer found in release {version}.")
        print(f"    Download manually from: {DOWNLOAD_PAGE}")
        return None

    tmp_dir   = Path(os.environ.get("TEMP", os.environ.get("TMP", "/tmp")))
    installer = tmp_dir / asset["name"]

    size_mb = asset["size"] / 1e6
    print(f"📦  Vortelio {version}  —  {asset['name']}  ({size_mb:.1f} MB)")
    print(f"    Downloading...")

    try:
        _download(asset["browser_download_url"], installer)
    except Exception as e:
        print(f"❌  Download failed: {e}")
        return None

    print("⚙️   Installing Vortelio (silent)...")
    try:
        # /S = silent install (NSIS flag)
        result = subprocess.run(
            [str(installer), "/S"],
            timeout=120,
        )
        installer.unlink(missing_ok=True)
        if result.returncode != 0:
            print(f"❌  Installer exited with code {result.returncode}")
            return None
    except subprocess.TimeoutExpired:
        print("❌  Installer timed out.")
        return None
    except Exception as e:
        print(f"❌  Installer failed: {e}")
        return None

    # Wait a moment for files to be written
    time.sleep(3)

    # Find the newly installed exe
    exe = find_vortelio_exe()
    if exe:
        print(f"✅  Vortelio installed: {exe}")
    else:
        print("⚠️   Installed but could not locate vortelio.exe")
        print(f"    Try: vortelio serve")
    return exe


# ── Main entry point ──────────────────────────────────────────────────────────

def ensure_server(port: int = 11500) -> bool:
    """
    Guarantee the Vortelio server is running on `port`.

    Flow:
      1. Server already running?  → done ✓
      2. vortelio.exe found?        → start it → done ✓
      3. Not found?               → prompt install → install → start → done ✓
    """
    # Step 1: already running
    if is_server_running(port):
        return True

    # Step 2: exe exists but server not running → start it
    exe = find_vortelio_exe()
    if exe:
        return start_server(exe, port)

    # Step 3: not installed → show prompt
    if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
        # Headless Linux / cloud
        print()
        print("=" * 58)
        print("  Vortelio SDK — server not found")
        print("=" * 58)
        print("  Running on a cloud/server environment.")
        print("  Install Vortelio on your LOCAL machine:")
        print(f"  {DOWNLOAD_PAGE}")
        print()
        print("  Then connect with:")
        print("  ai = Vortelio(auto_install=False)")
        return False

    print()
    print("+" + "-" * 56 + "+")
    print("|" + "  Vortelio is not installed on this PC".center(56) + "|")
    print("+" + "-" * 56 + "+")
    print("|  Vortelio is needed to run AI models locally.          |")
    print("|  The installer will be downloaded and run silently.  |")
    print("+" + "-" * 56 + "+")
    print()

    try:
        answer = input("  Install Vortelio now? [Y/n]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\n  Skipped.")
        return False

    if answer not in ("", "y", "yes"):
        print(f"\n  Download manually from: {DOWNLOAD_PAGE}")
        return False

    exe = install_vortelio()
    if not exe:
        return False

    return start_server(exe, port)
