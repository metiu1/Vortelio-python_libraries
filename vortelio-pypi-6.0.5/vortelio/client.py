"""
Vortelio Python SDK
Requires the server: vortelio serve
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Callable, Generator, Iterator, List, Optional


class Vortelio:
    """
    Python client for Vortelio — run AI models locally.

    Start the server first with:  vortelio serve

    Example::

        from vortelio import Vortelio

        ai = Vortelio()
        ai.models()
        ai.chat("llm/mistral:7b", "Hello!")
        ai.image("image/sdxl", "a cat in space", "cat.png")
    """

    def __init__(self, port: int = 11500, auto_install: bool = True) -> None:
        """
        Create the Vortelio client.

        On first use, automatically checks if the Vortelio server is installed
        and running. If not installed, asks the user whether to install it.
        If installed but not running, starts it automatically.

        Args:
            port         (int):  Port of the Vortelio server. Default 11500.
            auto_install (bool): If True (default), prompt to install Vortelio
                                 if not found. Set to False to skip the check.

        Example::

            ai = Vortelio()                     # auto-check and install if needed
            ai = Vortelio(auto_install=False)   # skip install check
            ai = Vortelio(port=8080)            # custom port
        """
        self._base: str = f"http://127.0.0.1:{port}"
        if auto_install:
            from .setup import ensure_server
            ok = ensure_server(port=port)
            if not ok:
                print()
                print("⚠️   Vortelio server is not running.")
                print(f"    Open a new terminal and run:  vortelio serve")
                print(f"    Then re-run your script.")
                print()

    # ── STATUS ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """
        Return server status: version, hardware, number of models.

        Returns:
            dict: ``{"version": "0.3.x", "hardware": "CUDA …", "model_count": 3}``

        Example::

            info = ai.status()
            print(info["hardware"])  # "CUDA (GPU 0: RTX 3080, 10 GB VRAM)"
        """
        return self._get("/api/status")

    # ── MODELS ────────────────────────────────────────────────────────────

    def models(self) -> List[dict]:
        """
        Return a list of all locally installed models and print a summary table.

        Returns:
            List[dict]: Each dict has ``type``, ``name``, ``tag``, ``format``,
                        ``size_bytes``, ``size_human``.

        Example::

            for m in ai.models():
                print(m["type"], m["name"], m["size_human"])
        """
        response: dict = self._get("/api/models")
        items: List[dict] = response.get("models") or []

        if not items:
            print("No models installed.")
            print("Download one with:  vortelio pull llm/mistral:7b")
            return []

        print(f"\n📦  {len(items)} model{'s' if len(items) != 1 else ''} installed")
        print("─" * 58)
        for i, m in enumerate(items, 1):
            ref   = f"{m.get('type','?')}/{m.get('name','?')}:{m.get('tag','?')}"
            size  = m.get("size_human") or _human_size(m.get("size_bytes", 0))
            fmt   = m.get("format", "")
            dname = m.get("display_name", "")
            label = f"{ref}" + (f"  ({dname})" if dname else "")
            print(f"  {i:<3} {label:<38} {size:>8}   {fmt}")
        print()
        return items

    def pull(self, model: str, on_progress: Optional[Callable[[int, str], None]] = None) -> None:
        """
        Download a model from HuggingFace via the Vortelio server.

        The type prefix is required: ``llm/``, ``image/``, ``audio/``,
        ``video/``, or ``3d/``.

        Args:
            model       (str):      Model reference.
                                    Examples: ``"llm/mistral:7b"``
                                              ``"image/sdxl:latest"``
                                              ``"llm/hf.co/owner/repo:file.gguf"``
            on_progress (callable): Optional callback ``fn(pct: int, msg: str)``.
                                    Called with download percentage and message.

        Example::

            ai.pull("llm/mistral:7b")
            ai.pull("image/flux:schnell", on_progress=lambda p, m: print(f"{p}% {m}"))
        """
        body = json.dumps({"model": model}).encode()
        req  = urllib.request.Request(
            f"{self._base}/api/pull", data=body,
            method="POST", headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                _parse_sse(resp,
                    on_progress=on_progress or (lambda p, m: print(f"\r  {p:3d}%  {m}   ", end="", flush=True)),
                    on_done=lambda d: print(f"\n✅  Downloaded: {model}"),
                    on_error=lambda e: (_ for _ in ()).throw(RuntimeError(e)),
                )
            print()
        except urllib.error.URLError as exc:
            raise ConnectionError(_conn_error(self._base)) from exc

    # ── CHAT ─────────────────────────────────────────────────────────────

    def chat(
        self,
        model: str,
        message: str,
        *,
        context_size: int = 4096,
        think: bool = False,
        history: Optional[List[dict]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        silent: bool = False,
    ) -> str:
        """
        Send a message to a language model and return the reply.

        Tokens are streamed in real time — either printed to the terminal
        or passed one-by-one to your ``on_token`` callback.

        Args:
            model        (str):             Model to use, e.g. ``"llm/mistral:7b"``.
            message      (str):             The text message.
            context_size (int):             Max context tokens. Default 4096.
            think        (bool):            Enable chain-of-thought reasoning.
                                            The model's ``<think_>`` blocks are hidden
                                            from the terminal but available in the
                                            ``thinking`` attribute of the return value.
            history      (list, optional):  Previous messages for multi-turn context.
                                            Each item: ``{"role": "user"/"assistant",
                                            "content": "…"}``.
            on_token     (callable, optional): Called with each token string as it
                                            arrives. Disables automatic printing.
                                            Signature: ``fn(token: str) -> None``
            silent       (bool):            If True, suppress all terminal output.
                                            Useful when using ``on_token``.

        Returns:
            str: The complete reply as a plain string (excluding ``<think_>`` blocks).

        Example::

            # Default — tokens printed in real time
            reply = ai.chat("llm/mistral:7b", "What is Python?")

            # Custom token callback
            tokens = []
            ai.chat("llm/mistral:7b", "Hello!",
                    on_token=lambda t: tokens.append(t))

            # Stream to a file
            with open("output.txt", "w") as f:
                ai.chat("llm/mistral:7b", "Write a poem",
                        on_token=lambda t: f.write(t))

            # Websocket / HTTP streaming in a web app
            def send_to_ws(token):
                websocket.send(json.dumps({"token": token}))

            ai.chat("llm/mistral:7b", "Tell me a story", on_token=send_to_ws)

            # Silent — no output, just return the string
            reply = ai.chat("llm/mistral:7b", "Hello!", silent=True)
        """
        msgs = list(history or [])
        msgs.append({"role": "user", "content": message})
        return self._stream_llm(model, message, messages=msgs,
                                context_size=context_size, think=think,
                                on_token=on_token, silent=silent)

    def conversation(
        self,
        model: str,
        system: str = "You are a helpful assistant.",
    ) -> "Conversation":
        """
        Start a multi-turn conversation where the model remembers history.

        Args:
            model  (str): Model to use.
            system (str): System prompt. Default: ``"You are a helpful assistant."``.

        Returns:
            Conversation: Use ``.say()`` to send messages.

        Example::

            conv = ai.conversation("llm/mistral:7b")
            conv.say("My name is Marco.")
            conv.say("What is my name?")   # → "Your name is Marco."
            conv.start()                   # interactive REPL
        """
        return Conversation(model=model, system=system, client=self)

    # ── IMAGE ────────────────────────────────────────────────────────────

    def image(
        self,
        model: str,
        description: str,
        save_to: str,
        steps: int = 20,
    ) -> str:
        """
        Generate an image from a text description and save it to a file.

        Args:
            model       (str): Model, e.g. ``"image/sdxl"``, ``"image/flux:schnell"``.
            description (str): What to generate.
            save_to     (str): Output path, e.g. ``"output.png"``.
            steps       (int): Inference steps. More = better quality, slower.

        Returns:
            str: Path to the saved image.

        Example::

            ai.image("image/sdxl", "a purple sunset over the ocean", "sunset.png")
            ai.image("image/flux:schnell", "medieval castle", "castle.png", steps=30)
        """
        return self._generate_media(model, prompt=description, output=save_to, steps=steps)

    # ── AUDIO ────────────────────────────────────────────────────────────

    def transcribe(
        self,
        model: str,
        audio_file: str,
        save_to: Optional[str] = None,
    ) -> str:
        """
        Transcribe an audio file to text (Speech-to-Text) using Whisper.

        Args:
            model      (str):  Whisper model, e.g. ``"audio/whisper:large"``.
            audio_file (str):  Path to ``.mp3``/``.wav``/``.flac``/``.m4a`` file.
            save_to    (str):  Optional path to save the transcript as ``.txt``.

        Returns:
            str: Transcribed text.

        Example::

            text = ai.transcribe("audio/whisper:large", "meeting.mp3")
            ai.transcribe("audio/whisper:base", "note.wav", save_to="note.txt")
        """
        return self._generate_media(model, input_file=audio_file, output=save_to)

    def speak(
        self,
        model: str,
        text: str,
        save_to: str,
    ) -> str:
        """
        Convert text to speech and save as ``.wav``.

        Args:
            model   (str): TTS model, e.g. ``"audio/kokoro"`` or ``"audio/bark"``.
            text    (str): Text to speak.
            save_to (str): Output ``.wav`` path.

        Returns:
            str: Path to the saved audio file.

        Example::

            ai.speak("audio/kokoro", "Hello! I am Vortelio.", "hello.wav")
        """
        return self._generate_media(model, prompt=text, output=save_to)

    # ── VIDEO ────────────────────────────────────────────────────────────

    def video(
        self,
        model: str,
        description: str,
        save_to: str,
        steps: int = 20,
    ) -> str:
        """
        Generate a video from a text description and save as ``.mp4``.

        Args:
            model       (str): Video model, e.g. ``"video/wan:1.3b"``.
            description (str): What the video should show.
            save_to     (str): Output ``.mp4`` path.
            steps       (int): Generation steps.

        Returns:
            str: Path to the saved video.

        Example::

            ai.video("video/wan:1.3b", "a flying cat", "cat.mp4")
        """
        return self._generate_media(model, prompt=description, output=save_to, steps=steps)

    # ── 3D ───────────────────────────────────────────────────────────────

    def model3d(
        self,
        model: str,
        save_to: str,
        description: Optional[str] = None,
        image: Optional[str] = None,
    ) -> str:
        """
        Generate a 3D model from text or from an image.

        Args:
            model       (str):  3D model, e.g. ``"3d/triposr"`` or ``"3d/shap-e"``.
            save_to     (str):  Output path, e.g. ``"chair.obj"``.
            description (str):  Text description.
            image       (str):  Path to input image (for image→3D).

        Returns:
            str: Path to the saved 3D file.

        Example::

            ai.model3d("3d/shap-e",  "chair.ply", description="a wooden chair")
            ai.model3d("3d/triposr", "chair.obj", image="photo.jpg")
        """
        if not description and not image:
            raise ValueError("Provide at least 'description' or 'image'.")
        return self._generate_media(
            model, prompt=description or "", input_file=image, output=save_to
        )

    # ── INTERNALS ────────────────────────────────────────────────────────

    def _stream_llm(
        self,
        model: str,
        prompt: str,
        messages: Optional[List[dict]] = None,
        context_size: int = 4096,
        think: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
        silent: bool = False,
    ) -> str:
        """Internal: stream LLM tokens from the Vortelio server."""
        body = json.dumps({
            "model":        model,
            "prompt":       prompt,
            "stream":       True,
            "messages":     messages or [],
            "context_size": context_size,
            "think":        think,
        }).encode()
        req = urllib.request.Request(
            f"{self._base}/api/generate", data=body,
            method="POST", headers={"Content-Type": "application/json"},
        )

        reply_tokens:  List[str] = []   # visible reply tokens
        think_tokens:  List[str] = []   # tokens inside <think_>...</think_>
        buf:           str       = ""   # line buffer for SSE parsing
        in_think:      bool      = False
        think_buf:     str       = ""   # accumulates across tokens

        def _emit(token: str) -> None:
            """Handle one token from the stream."""
            nonlocal in_think, think_buf

            think_buf += token

            # State machine: detect and route <think_>...</think_> blocks
            while think_buf:
                if in_think:
                    end_idx = think_buf.find("</think_>")
                    if end_idx >= 0:
                        # Collect the think content
                        think_tokens.append(think_buf[:end_idx])
                        think_buf = think_buf[end_idx + 9:]
                        in_think = False
                    else:
                        # Still inside think block — buffer everything
                        think_tokens.append(think_buf)
                        think_buf = ""
                else:
                    start_idx = think_buf.find("<think_>")
                    if start_idx >= 0:
                        # Emit everything before <think_>
                        visible = think_buf[:start_idx]
                        think_buf = think_buf[start_idx + 8:]
                        in_think = True
                        if visible:
                            reply_tokens.append(visible)
                            if on_token:
                                on_token(visible)
                            elif not silent:
                                print(visible, end="", flush=True)
                    else:
                        # All visible — emit immediately
                        reply_tokens.append(think_buf)
                        if on_token:
                            on_token(think_buf)
                        elif not silent:
                            print(think_buf, end="", flush=True)
                        think_buf = ""
                        break

        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                # Read raw bytes and split on newlines for SSE
                raw_buf = b""
                while True:
                    chunk = resp.read(512)
                    if not chunk:
                        break
                    raw_buf += chunk
                    # Process all complete lines
                    while b"\n" in raw_buf:
                        line_bytes, raw_buf = raw_buf.split(b"\n", 1)
                        line = line_bytes.decode("utf-8", errors="replace").rstrip("\r")
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            raw_buf = b""  # signal done
                            break
                        if payload.startswith("[ERROR]"):
                            raise RuntimeError(payload[8:].strip())
                        _emit(payload)

        except urllib.error.URLError as exc:
            raise ConnectionError(_conn_error(self._base)) from exc

        if not silent and not on_token:
            print()   # newline after streaming

        return "".join(reply_tokens)

    def stream(
        self,
        model: str,
        message: str,
        *,
        context_size: int = 4096,
        think: bool = False,
        history: Optional[List[dict]] = None,
    ) -> Iterator[str]:
        """
        Iterate over tokens as they arrive — generator interface.

        Each ``yield`` gives you one token string the moment it's generated.
        Useful for web frameworks, async pipelines, or custom UIs.

        Args:
            model        (str):  Model to use.
            message      (str):  Your message.
            context_size (int):  Context window size. Default 4096.
            think        (bool): Enable chain-of-thought reasoning.
            history      (list): Conversation history.

        Yields:
            str: One token at a time, as soon as it arrives.

        Example::

            # Print each token as it arrives (manual control)
            for token in ai.stream("llm/mistral:7b", "Tell me a story"):
                print(token, end="", flush=True)
            print()

            # Collect all tokens
            tokens = list(ai.stream("llm/mistral:7b", "Hello!"))
            reply  = "".join(tokens)

            # Pipe to Flask streaming response
            from flask import Response, stream_with_context

            @app.route("/chat")
            def chat():
                def generate():
                    for token in ai.stream("llm/mistral:7b", request.args["q"]):
                        yield f"data: {json.dumps(token)}\n\n"
                    yield "data: [DONE]\n\n"
                return Response(stream_with_context(generate()),
                                content_type="text/event-stream")

            # FastAPI / async (run in executor)
            import asyncio
            from fastapi import FastAPI
            from fastapi.responses import StreamingResponse

            app = FastAPI()

            @app.get("/chat")
            async def chat(q: str):
                def gen():
                    for token in ai.stream("llm/mistral:7b", q):
                        yield f"data: {token}\n\n"
                return StreamingResponse(gen(), media_type="text/event-stream")
        """
        msgs = list(history or [])
        msgs.append({"role": "user", "content": message})

        body = json.dumps({
            "model":        model,
            "prompt":       message,
            "stream":       True,
            "messages":     msgs,
            "context_size": context_size,
            "think":        think,
        }).encode()
        req = urllib.request.Request(
            f"{self._base}/api/generate", data=body,
            method="POST", headers={"Content-Type": "application/json"},
        )

        in_think  = False
        think_buf = ""

        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                raw_buf = b""
                while True:
                    chunk = resp.read(512)
                    if not chunk:
                        break
                    raw_buf += chunk
                    while b"\n" in raw_buf:
                        line_bytes, raw_buf = raw_buf.split(b"\n", 1)
                        line = line_bytes.decode("utf-8", errors="replace").rstrip("\r")
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            return
                        if payload.startswith("[ERROR]"):
                            raise RuntimeError(payload[8:].strip())

                        # Route through think filter
                        think_buf += payload
                        while think_buf:
                            if in_think:
                                end_idx = think_buf.find("</think_>")
                                if end_idx >= 0:
                                    think_buf = think_buf[end_idx + 9:]
                                    in_think = False
                                else:
                                    think_buf = ""
                            else:
                                start_idx = think_buf.find("<think_>")
                                if start_idx >= 0:
                                    visible = think_buf[:start_idx]
                                    think_buf = think_buf[start_idx + 8:]
                                    in_think = True
                                    if visible:
                                        yield visible
                                else:
                                    yield think_buf
                                    think_buf = ""
                                    break
        except urllib.error.URLError as exc:
            raise ConnectionError(_conn_error(self._base)) from exc

    def _generate_media(
        self,
        model: str,
        prompt: str = "",
        input_file: Optional[str] = None,
        output: Optional[str] = None,
        steps: int = 20,
    ) -> str:
        body = json.dumps({
            "model":       model,
            "prompt":      prompt,
            "input_file":  str(input_file) if input_file else "",
            "output_file": str(output) if output else "",
            "steps":       steps,
        }).encode()
        req = urllib.request.Request(
            f"{self._base}/api/generate", data=body,
            method="POST", headers={"Content-Type": "application/json"},
        )
        result = {"saved_to": "", "data": None, "error": None}

        def _on_progress(pct: int, msg: str) -> None:
            print(f"\r  {pct:3d}%  {msg}   ", end="", flush=True)

        def _on_result(ev: dict) -> None:
            result["saved_to"] = ev.get("saved_to", "")
            result["data"]     = ev.get("data")
            if result["saved_to"]:
                print(f"\n✅  Saved to: {result['saved_to']}")

        def _on_error(msg: str) -> None:
            result["error"] = msg

        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                _parse_sse(resp,
                    on_progress=_on_progress,
                    on_done=None,
                    on_error=_on_error,
                    on_result=_on_result,
                )
            print()
        except urllib.error.HTTPError as exc:
            msg = json.loads(exc.read()).get("error", str(exc))
            raise RuntimeError(msg) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(_conn_error(self._base)) from exc

        if result["error"]:
            raise RuntimeError(result["error"])

        # Save base64 data if server returned it directly
        if result["data"] and output:
            Path(output).write_bytes(base64.b64decode(result["data"]))
            return str(output)

        return result["saved_to"] or ""

    def _get(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(f"{self._base}{path}", timeout=10) as r:
                return json.loads(r.read())
        except urllib.error.URLError as exc:
            raise ConnectionError(_conn_error(self._base)) from exc

    def __repr__(self) -> str:
        return f"<Vortelio {self._base}>"


# ── CONVERSATION ─────────────────────────────────────────────────────────────

class Conversation:
    """
    Multi-turn conversation — the model remembers all previous messages.

    Do not instantiate directly: use ``ai.conversation()`` instead.

    Example::

        conv = ai.conversation("llm/mistral:7b")
        conv.say("My name is Marco.")
        conv.say("What is my name?")   # → "Your name is Marco."
        conv.clear()
        conv.start()                   # interactive REPL
    """

    def __init__(self, model: str, system: str, client: "Vortelio") -> None:
        self._model:   str        = model
        self._client:  Vortelio   = client
        self._history: List[dict] = [{"role": "system", "content": system}]

    def say(
        self,
        message: str,
        *,
        think: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
        silent: bool = False,
    ) -> str:
        """
        Send a message and get a reply with full history context.

        Tokens are streamed in real time — printed, passed to ``on_token``,
        or suppressed with ``silent=True``.

        Args:
            message  (str):              Your message.
            think    (bool):             Enable chain-of-thought reasoning.
            on_token (callable, optional): Called with each token as it arrives.
                                         Signature: ``fn(token: str) -> None``
            silent   (bool):             Suppress all terminal output.

        Returns:
            str: The model's full reply.

        Example::

            conv = ai.conversation("llm/mistral:7b")

            # Default — streams to terminal
            conv.say("What is Python?")

            # Custom token handler
            conv.say("Tell me a story",
                     on_token=lambda t: my_ui.append(t))

            # Silent — just return string
            reply = conv.say("Hello!", silent=True)
        """
        self._history.append({"role": "user", "content": message})
        reply = self._client._stream_llm(
            self._model, message,
            messages=self._history,
            think=think,
            on_token=on_token,
            silent=silent,
        )
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def stream(
        self,
        message: str,
        *,
        think: bool = False,
    ) -> "Iterator[str]":
        """
        Send a message and iterate over tokens as they arrive.

        The history is updated automatically after all tokens are received.

        Args:
            message (str):  Your message.
            think   (bool): Enable chain-of-thought reasoning.

        Yields:
            str: One token at a time.

        Example::

            conv = ai.conversation("llm/mistral:7b")
            for token in conv.stream("Explain recursion"):
                print(token, end="", flush=True)
            print()
        """
        self._history.append({"role": "user", "content": message})
        tokens: List[str] = []
        for token in self._client.stream(
            self._model, message,
            history=self._history[:-1],  # exclude the just-added user msg
            think=think,
        ):
            tokens.append(token)
            yield token
        reply = "".join(tokens)
        self._history.append({"role": "assistant", "content": reply})

    def start(self) -> None:
        """
        Start an interactive terminal REPL.
        Type ``exit`` to quit, ``clear`` to reset memory.
        """
        print(f"\n💬  Conversation with {self._model}")
        print("    Type 'exit' to quit · 'clear' to reset memory\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n👋  Goodbye!")
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("👋  Goodbye!")
                break
            if user_input.lower() == "clear":
                self.clear()
                print("🗑   Memory cleared.\n")
                continue
            print("AI: ", end="", flush=True)
            self.say(user_input)
            print()

    def clear(self) -> None:
        """Reset history, keeping the system prompt."""
        self._history = [self._history[0]]

    def history(self) -> List[dict]:
        """Return conversation history as a list of ``{role, content}`` dicts."""
        return list(self._history)

    def __repr__(self) -> str:
        turns = (len(self._history) - 1) // 2
        return f"<Conversation model={self._model!r} turns={turns}>"


# ── HELPERS ──────────────────────────────────────────────────────────────────

def _parse_sse(
    resp,
    on_progress: Optional[Callable] = None,
    on_done:     Optional[Callable] = None,
    on_error:    Optional[Callable] = None,
    on_result:   Optional[Callable] = None,
) -> None:
    """Parse a Server-Sent Events stream from Vortelio server."""
    buf = ""
    for raw in resp:
        buf += raw.decode("utf-8", errors="replace")
        while "\n\n" in buf:
            chunk, buf = buf.split("\n\n", 1)
            etype, dline = "", ""
            for line in chunk.split("\n"):
                if line.startswith("event: "):
                    etype = line[7:].strip()
                elif line.startswith("data: "):
                    dline = line[6:].strip()
            if not dline:
                continue
            try:
                ev = json.loads(dline)
            except json.JSONDecodeError:
                continue
            if etype == "progress" and on_progress:
                on_progress(ev.get("pct", 0), ev.get("msg", ""))
            elif etype == "result" and on_result:
                on_result(ev)
            elif etype == "done" and on_done:
                on_done(ev)
            elif etype == "error" and on_error:
                on_error(ev.get("error", "Unknown error"))


def _conn_error(base: str) -> str:
    return (
        f"Vortelio server not reachable at {base}\n"
        "Start it with:  vortelio serve\n"
        "Or open the GUI: vortelio gui"
    )


def _human_size(b: int) -> str:
    if b >= 1_000_000_000:
        return f"{b / 1e9:.1f} GB"
    if b >= 1_000_000:
        return f"{b / 1e6:.0f} MB"
    return f"{b / 1e3:.0f} KB"
