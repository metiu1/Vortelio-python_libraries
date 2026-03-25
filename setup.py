"""
PullAI — Python SDK for running AI models locally.

On first use, automatically detects if PullAI is installed and running.
If not installed, offers to download and install it automatically.

Quick start::

    from pullai import PullAI

    ai = PullAI()           # auto-installs server if needed
    ai.models()
    ai.pull("llm/mistral:7b")
    ai.chat("llm/mistral:7b", "Hello!")
    ai.image("image/sdxl", "a sunset on Mars", "mars.png")
"""

from .client import PullAI, Conversation
from .setup  import find_pullai_exe, is_server_running, ensure_server, install_pullai

__version__ = "6.0.5"
__author__  = "Metiu"
__all__     = [
    "PullAI",
    "Conversation",
    "find_pullai_exe",
    "is_server_running",
    "ensure_server",
    "install_pullai",
]
