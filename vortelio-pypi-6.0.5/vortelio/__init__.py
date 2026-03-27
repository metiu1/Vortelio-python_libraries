"""
Vortelio — Python SDK for running AI models locally.

On first use, automatically detects if Vortelio is installed and running.
If not installed, offers to download and install it automatically.

Quick start::

    from vortelio import Vortelio

    ai = Vortelio()           # auto-installs server if needed
    ai.models()
    ai.pull("llm/mistral:7b")
    ai.chat("llm/mistral:7b", "Hello!")
    ai.image("image/sdxl", "a sunset on Mars", "mars.png")
"""

from .client import Vortelio, Conversation
from .setup  import find_vortelio_exe, is_server_running, ensure_server, install_vortelio

__version__ = "6.0.5"
__author__  = "Metiu"
__all__     = [
    "Vortelio",
    "Conversation",
    "find_vortelio_exe",
    "is_server_running",
    "ensure_server",
    "install_vortelio",
]
