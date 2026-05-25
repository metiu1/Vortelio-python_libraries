"""Vortelio synchronous Python client."""
from __future__ import annotations

import base64
import json
import os
import sys
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Union

from ._http import VortElioError, _request, _stream_ndjson, _stream_sse


class Conversation:
    """Stateful multi-turn conversation helper."""

    def __init__(self, client: "Vortelio", model: str, system: Optional[str] = None) -> None:
        self._client = client
        self._model = model
        self._history: List[Dict[str, Any]] = []
        if system:
            self._history.append({"role": "system", "content": system})

    def say(
        self,
        prompt: str,
        *,
        on_token: Optional[Callable[[str], None]] = None,
        silent: bool = False,
        **kwargs: Any,
    ) -> str:
        self._history.append({"role": "user", "content": prompt})
        reply_parts: List[str] = []

        def _collect(tok: str) -> None:
            reply_parts.append(tok)
            if not silent:
                print(tok, end="", flush=True)
            if on_token:
                on_token(tok)

        self._client.chat(self._model, self._history, on_token=_collect, silent=True, **kwargs)
        reply = "".join(reply_parts)
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        self._history.append({"role": "user", "content": prompt})
        reply_parts: List[str] = []

        for chunk in self._client.chat_stream(self._model, self._history, **kwargs):
            reply_parts.append(chunk)
            yield chunk

        self._history.append({"role": "assistant", "content": "".join(reply_parts)})

    def reset(self, keep_system: bool = True) -> None:
        if keep_system and self._history and self._history[0].get("role") == "system":
            self._history = [self._history[0]]
        else:
            self._history = []

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)


class Vortelio:
    """
    Synchronous Python client for the Vortelio local AI server.

    Quick start::

        from vortelio import Vortelio

        ai = Vortelio()               # default: http://localhost:11500
        ai.pull("llm/mistral:7b")
        ai.chat("llm/mistral:7b", "Hello, how are you?")

        # Streaming
        for token in ai.chat_stream("llm/mistral:7b", "Tell me a story"):
            print(token, end="", flush=True)

        # OpenAI-compat
        resp = ai.openai_chat("mistral:7b", [{"role":"user","content":"Hi"}])
    """

    def __init__(
        self,
        host: str = "http://localhost",
        port: int = 11500,
        timeout: int = 300,
    ) -> None:
        self._base = f"{host.rstrip('/')}:{port}"
        self._timeout = timeout

    def _url(self, path: str) -> str:
        return self._base + path

    def _get(self, path: str) -> Dict[str, Any]:
        return _request("GET", self._url(path), timeout=self._timeout)

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return _request("POST", self._url(path), body=body, timeout=self._timeout)

    def _delete(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return _request("DELETE", self._url(path), body=body, timeout=self._timeout)

    def _patch(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return _request("PATCH", self._url(path), body=body, timeout=self._timeout)

    # ── Server info ───────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Server health, version, and hardware info."""
        return self._get("/api/status")

    def version(self) -> str:
        """Vortelio server version string."""
        return self._get("/api/version").get("version", "")

    def ps(self) -> List[Dict[str, Any]]:
        """Currently loaded models."""
        return self._get("/api/ps").get("models", [])

    # ── Model management ──────────────────────────────────────────────────────

    def models(self) -> List[Dict[str, Any]]:
        """List all downloaded models with sizes."""
        result = self._get("/api/models")
        return result.get("models", [])

    def tags(self) -> List[Dict[str, Any]]:
        """Ollama-compatible model list."""
        return self._get("/api/tags").get("models", [])

    def show(self, model: str) -> Dict[str, Any]:
        """Show model metadata: modelfile, template, parameters, capabilities."""
        return self._post("/api/show", {"model": model})

    def pull(
        self,
        model: str,
        *,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        silent: bool = False,
    ) -> None:
        """Download a model from HuggingFace. Streams progress events."""
        body = {"model": model, "stream": True}
        for evt in _stream_sse(self._url("/api/pull"), body, timeout=7200):
            event = evt.get("_event", "message")
            if event == "progress":
                if not silent:
                    msg = evt.get("msg", "")
                    pct = evt.get("pct", 0)
                    print(f"\r  {model}: {pct:3d}%  {msg}   ", end="", flush=True)
                if on_progress:
                    on_progress(evt)
            elif event == "done":
                if not silent:
                    print(f"\r  {model}: done{' ' * 40}")
                return
            elif event == "error":
                raise VortElioError(500, evt.get("error", "download failed"))

    def pull_cancel(self, model: str) -> Dict[str, Any]:
        """Cancel an in-progress download."""
        return self._post("/api/pull/cancel", {"model": model})

    def delete(self, model: str) -> None:
        """Delete a downloaded model."""
        self._delete("/api/delete", {"model": model})

    def copy(self, source: str, destination: str) -> None:
        """Copy/duplicate a model under a new name."""
        self._post("/api/copy", {"source": source, "destination": destination})

    def create(
        self,
        model: str,
        *,
        modelfile: Optional[str] = None,
        from_model: Optional[str] = None,
        system: Optional[str] = None,
        quantize: Optional[str] = None,
        on_status: Optional[Callable[[str], None]] = None,
        silent: bool = False,
    ) -> None:
        """Create a model from a Modelfile or simple options."""
        body: Dict[str, Any] = {"model": model, "stream": True}
        if modelfile:
            body["modelfile"] = modelfile
        if from_model:
            body["from"] = from_model
        if system:
            body["system"] = system
        if quantize:
            body["quantize"] = quantize

        for chunk in _stream_ndjson(self._url("/api/create"), body, timeout=7200):
            status = chunk.get("status", "")
            if not silent:
                print(f"  create: {status}")
            if on_status:
                on_status(status)

    def quantize(
        self,
        model: str,
        quantize: str,
        *,
        output: Optional[str] = None,
        on_status: Optional[Callable[[str], None]] = None,
        silent: bool = False,
    ) -> None:
        """Quantize a model (requires llama-quantize in PATH)."""
        body: Dict[str, Any] = {"model": model, "quantize": quantize, "stream": True}
        if output:
            body["output"] = output

        for chunk in _stream_ndjson(self._url("/api/quantize"), body, timeout=7200):
            status = chunk.get("status", "")
            if not silent:
                print(f"  quantize: {status}")
            if on_status:
                on_status(status)
            if chunk.get("error"):
                raise VortElioError(500, chunk["error"])

    def rename(self, model: str, display_name: str) -> Dict[str, Any]:
        """Set a display name for a model."""
        return self._post("/api/models/rename", {"model": model, "display_name": display_name})

    def set_mmproj(self, model: str, mmproj_path: str) -> Dict[str, Any]:
        """Set the multimodal projector (llava-style) for a model."""
        return self._post("/api/models/mmproj", {"model": model, "mmproj_path": mmproj_path})

    # ── Blobs ─────────────────────────────────────────────────────────────────

    def blob_exists(self, digest: str) -> bool:
        """Check if a blob exists by sha256 digest."""
        import urllib.error
        import urllib.request
        req = urllib.request.Request(
            self._url(f"/api/blobs/{digest}"),
            method="HEAD",
        )
        try:
            urllib.request.urlopen(req, timeout=self._timeout)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            raise

    def blob_upload(self, digest: str, data: bytes) -> None:
        """Upload a blob. digest must be sha256:<64 hex chars>."""
        import hashlib
        actual = "sha256:" + hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise ValueError(f"digest mismatch: expected {digest}, got {actual}")
        import urllib.request
        req = urllib.request.Request(
            self._url(f"/api/blobs/{digest}"),
            data=data,
            method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        urllib.request.urlopen(req, timeout=self._timeout)

    # ── LLM — generate ────────────────────────────────────────────────────────

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[List[str]] = None,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
        think: bool = False,
        raw: bool = False,
        context: Optional[List[int]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        silent: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate text with /api/generate (non-streaming result).
        Returns the full response dict including 'response', 'context', 'thinking'.
        """
        body: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if system:
            body["system"] = system
        if images:
            body["images"] = images
        if format:
            body["format"] = format
        if options:
            body["options"] = options
        if keep_alive:
            body["keep_alive"] = keep_alive
        if think:
            body["think"] = True
        if raw:
            body["raw"] = True
        if context:
            body["context"] = context

        return self._post("/api/generate", body)

    def generate_stream(
        self,
        model: str,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[List[str]] = None,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
        think: bool = False,
        raw: bool = False,
        context: Optional[List[int]] = None,
    ) -> Generator[str, None, None]:
        """Stream tokens from /api/generate. Yields token strings."""
        body: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": True}
        if system:
            body["system"] = system
        if images:
            body["images"] = images
        if format:
            body["format"] = format
        if options:
            body["options"] = options
        if keep_alive:
            body["keep_alive"] = keep_alive
        if think:
            body["think"] = True
        if raw:
            body["raw"] = True
        if context:
            body["context"] = context

        for chunk in _stream_ndjson(self._url("/api/generate"), body, timeout=self._timeout):
            if chunk.get("done"):
                return
            tok = chunk.get("response", "")
            if tok:
                yield tok

    # ── LLM — chat ────────────────────────────────────────────────────────────

    def chat(
        self,
        model: str,
        messages: Union[List[Dict[str, Any]], str],
        *,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        think: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
        silent: bool = False,
    ) -> str:
        """
        Chat with streaming output printed in real time.
        'messages' can be a plain string (treated as single user message).
        Returns the complete assistant response as a string.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if format:
            body["format"] = format
        if options:
            body["options"] = options
        if keep_alive:
            body["keep_alive"] = keep_alive
        if tools:
            body["tools"] = tools
        if think:
            body["think"] = True

        parts: List[str] = []
        for chunk in _stream_ndjson(self._url("/api/chat"), body, timeout=self._timeout):
            if chunk.get("done"):
                break
            tok = chunk.get("message", {}).get("content", "")
            if tok:
                parts.append(tok)
                if not silent:
                    print(tok, end="", flush=True)
                if on_token:
                    on_token(tok)

        if not silent:
            print()
        return "".join(parts)

    def chat_stream(
        self,
        model: str,
        messages: Union[List[Dict[str, Any]], str],
        *,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        think: bool = False,
    ) -> Generator[str, None, None]:
        """Yield tokens from /api/chat as they arrive."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if format:
            body["format"] = format
        if options:
            body["options"] = options
        if keep_alive:
            body["keep_alive"] = keep_alive
        if tools:
            body["tools"] = tools
        if think:
            body["think"] = True

        for chunk in _stream_ndjson(self._url("/api/chat"), body, timeout=self._timeout):
            if chunk.get("done"):
                return
            tok = chunk.get("message", {}).get("content", "")
            if tok:
                yield tok

    def chat_raw(
        self,
        model: str,
        messages: Union[List[Dict[str, Any]], str],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Low-level /api/chat — returns the full response dict (non-streaming)."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": False, **kwargs}
        return self._post("/api/chat", body)

    def stream(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Alias: stream tokens from generate_stream."""
        return self.generate_stream(model, prompt, **kwargs)

    def conversation(
        self,
        model: str,
        *,
        system: Optional[str] = None,
    ) -> Conversation:
        """Create a stateful multi-turn conversation."""
        return Conversation(self, model, system=system)

    # ── Embeddings ────────────────────────────────────────────────────────────

    def embed(
        self,
        model: str,
        input: Union[str, List[str]],
        *,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> List[List[float]]:
        """Batch embeddings via /api/embed. Returns list of embedding vectors."""
        body: Dict[str, Any] = {"model": model, "input": input}
        if options:
            body["options"] = options
        if keep_alive:
            body["keep_alive"] = keep_alive
        return self._post("/api/embed", body).get("embeddings", [])

    def embeddings(
        self,
        model: str,
        prompt: str,
        *,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> List[float]:
        """Single-prompt legacy embedding via /api/embeddings."""
        body: Dict[str, Any] = {"model": model, "prompt": prompt}
        if options:
            body["options"] = options
        if keep_alive:
            body["keep_alive"] = keep_alive
        return self._post("/api/embeddings", body).get("embedding", [])

    # ── Advanced ─────────────────────────────────────────────────────────────

    def route(
        self,
        task: str,
        *,
        prompt: Optional[str] = None,
        min_params: Optional[str] = None,
        max_params: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Heuristic model router — picks the best local model for a task."""
        body: Dict[str, Any] = {"task": task}
        if prompt:
            body["prompt"] = prompt
        if min_params:
            body["min_params"] = min_params
        if max_params:
            body["max_params"] = max_params
        if capabilities:
            body["capabilities"] = capabilities
        return self._post("/api/route", body)

    def compare(
        self,
        models: List[str],
        prompt: str,
        *,
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """A/B compare multiple models on the same prompt. Returns all responses."""
        body: Dict[str, Any] = {"models": models, "prompt": prompt}
        if system:
            body["system"] = system
        return self._post("/api/compare", body)

    def structured(
        self,
        model: str,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        *,
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Force JSON output matching an optional JSON schema."""
        body: Dict[str, Any] = {"model": model, "prompt": prompt}
        if schema:
            body["schema"] = schema
        if system:
            body["system"] = system
        return self._post("/api/structured", body)

    def summarize(
        self,
        model: str,
        text: str,
        *,
        chunk_size: int = 8000,
        style: str = "paragraph",
    ) -> Dict[str, Any]:
        """
        Map-reduce summarization for long texts.
        style: "bullets" | "paragraph" | "tldr"
        """
        return self._post("/api/summarize", {
            "model": model,
            "text": text,
            "chunk_size": chunk_size,
            "style": style,
        })

    def think(
        self,
        model: str,
        prompt: str,
        *,
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Forced chain-of-thought. Returns {'thinking': ..., 'answer': ...}.
        Requires a model that supports <think> tags.
        """
        body: Dict[str, Any] = {"model": model, "prompt": prompt}
        if system:
            body["system"] = system
        return self._post("/api/think", body)

    # ── RAG ──────────────────────────────────────────────────────────────────

    def rag_ingest(
        self,
        model: str,
        documents: List[Dict[str, Any]],
        *,
        collection: str = "default",
        chunk_size: int = 800,
    ) -> Dict[str, Any]:
        """
        Add documents to a RAG collection and embed them.
        documents: list of {"text": "...", "meta": {...}}
        """
        return self._post("/api/rag/ingest", {
            "model": model,
            "collection": collection,
            "documents": documents,
            "chunk_size": chunk_size,
        })

    def rag_query(
        self,
        model: str,
        query: str,
        *,
        collection: str = "default",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Retrieve the top-k most similar chunks from a RAG collection."""
        return self._post("/api/rag/query", {
            "model": model,
            "collection": collection,
            "query": query,
            "top_k": top_k,
        })

    # ── Media generation ──────────────────────────────────────────────────────

    def generate_image(
        self,
        model: str,
        prompt: str,
        output_file: Optional[str] = None,
        *,
        steps: int = 20,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        silent: bool = False,
    ) -> bytes:
        """
        Generate an image. Returns raw PNG bytes.
        If output_file is given, also saves to disk.
        """
        return self._generate_media(
            model, prompt, output_file, steps=steps,
            on_progress=on_progress, silent=silent,
        )

    def generate_audio(
        self,
        model: str,
        prompt: str,
        output_file: Optional[str] = None,
        *,
        steps: int = 20,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        silent: bool = False,
    ) -> bytes:
        """Generate audio (TTS/music). Returns raw WAV bytes."""
        return self._generate_media(
            model, prompt, output_file, steps=steps,
            on_progress=on_progress, silent=silent,
        )

    def generate_video(
        self,
        model: str,
        prompt: str,
        output_file: Optional[str] = None,
        *,
        steps: int = 20,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        silent: bool = False,
    ) -> bytes:
        """Generate a video. Returns raw MP4 bytes."""
        return self._generate_media(
            model, prompt, output_file, steps=steps,
            on_progress=on_progress, silent=silent,
        )

    def generate_3d(
        self,
        model: str,
        prompt: str,
        output_file: Optional[str] = None,
        *,
        steps: int = 20,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        silent: bool = False,
    ) -> bytes:
        """Generate a 3D object. Returns raw OBJ bytes."""
        return self._generate_media(
            model, prompt, output_file, steps=steps,
            on_progress=on_progress, silent=silent,
        )

    def _generate_media(
        self,
        model: str,
        prompt: str,
        output_file: Optional[str],
        *,
        steps: int = 20,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        silent: bool = False,
    ) -> bytes:
        body: Dict[str, Any] = {"model": model, "prompt": prompt, "steps": steps, "stream": True}
        if output_file:
            body["output_file"] = output_file

        result_data: Optional[bytes] = None

        for evt in _stream_sse(self._url("/api/generate"), body, timeout=7200):
            event = evt.get("_event", "message")
            if event == "progress":
                if not silent:
                    pct = evt.get("pct", 0)
                    msg = evt.get("msg", "")
                    print(f"\r  {model}: {pct:3d}%  {msg}   ", end="", flush=True)
                if on_progress:
                    on_progress(evt)
            elif event == "result":
                if not silent:
                    print()
                b64 = evt.get("data", "")
                result_data = base64.b64decode(b64) if b64 else b""
                saved = evt.get("saved_to", "")
                if saved and not silent:
                    print(f"  saved to: {saved}")
            elif event == "error":
                raise VortElioError(500, evt.get("error", "generation failed"))

        return result_data or b""

    # ── Agents ────────────────────────────────────────────────────────────────

    def agents_catalog(self) -> List[Dict[str, Any]]:
        """List all installable AI agents (OpenClaw, Open WebUI, CrewAI, etc.)."""
        return self._get("/api/agents/catalog").get("agents", [])

    def agents_install(self, agent_id: str) -> Dict[str, Any]:
        """Install an agent by ID."""
        return self._post("/api/agents/install", {"agent": agent_id})

    def agents_start(self, agent_id: str) -> Dict[str, Any]:
        """Start an installed agent."""
        return self._post("/api/agents/start", {"agent": agent_id})

    def agents_stop(self, agent_id: str) -> Dict[str, Any]:
        """Stop a running agent."""
        return self._post("/api/agents/stop", {"agent": agent_id})

    def agents_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        return self._get("/api/agents/status")

    # ── Hooks (webhooks) ──────────────────────────────────────────────────────

    def hooks_list(self) -> List[Dict[str, Any]]:
        """List configured webhooks."""
        return self._get("/api/hooks").get("hooks", [])

    def hooks_create(
        self,
        url: str,
        *,
        event: str = "generate",
        secret: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a new webhook."""
        body: Dict[str, Any] = {"url": url, "event": event}
        if secret:
            body["secret"] = secret
        return self._post("/api/hooks", body)

    def hooks_delete(self, hook_id: str) -> Dict[str, Any]:
        """Delete a webhook by ID."""
        return self._delete("/api/hooks", {"id": hook_id})

    # ── Audit log ─────────────────────────────────────────────────────────────

    def audit(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Recent audit log entries."""
        return self._get(f"/api/audit?limit={limit}").get("entries", [])

    # ── GGUF inspect ─────────────────────────────────────────────────────────

    def gguf_inspect(self, path: str) -> Dict[str, Any]:
        """Parse GGUF file metadata from a local path."""
        return self._post("/api/gguf/inspect", {"path": path})

    # ── Import from Ollama ────────────────────────────────────────────────────

    def import_ollama(
        self,
        models: Optional[List[str]] = None,
        *,
        on_status: Optional[Callable[[str], None]] = None,
        silent: bool = False,
    ) -> Dict[str, Any]:
        """Import models from a locally installed Ollama instance."""
        body: Dict[str, Any] = {}
        if models:
            body["models"] = models
        result = self._post("/api/import/ollama", body)
        if not silent:
            imported = result.get("imported", [])
            for m in imported:
                print(f"  imported: {m}")
            if on_status:
                for m in imported:
                    on_status(m)
        return result

    # ── OpenAI-compatible endpoints ───────────────────────────────────────────

    def openai_chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """POST /v1/chat/completions — OpenAI-compatible (non-streaming)."""
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if seed is not None:
            body["seed"] = seed
        if tools:
            body["tools"] = tools
        if response_format:
            body["response_format"] = response_format
        return self._post("/v1/chat/completions", body)

    def openai_chat_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[str, None, None]:
        """Stream tokens from /v1/chat/completions (SSE)."""
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if tools:
            body["tools"] = tools

        for evt in _stream_sse(self._url("/v1/chat/completions"), body, timeout=self._timeout):
            choices = evt.get("choices", [])
            for choice in choices:
                delta = choice.get("delta", {})
                tok = delta.get("content", "")
                if tok:
                    yield tok

    def openai_completions(
        self,
        model: str,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """POST /v1/completions — OpenAI legacy completions."""
        body: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": stream}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        return self._post("/v1/completions", body)

    def openai_embeddings(
        self,
        model: str,
        input: Union[str, List[str]],
    ) -> Dict[str, Any]:
        """POST /v1/embeddings — OpenAI-compatible embeddings."""
        return self._post("/v1/embeddings", {"model": model, "input": input})

    def openai_models(self) -> List[Dict[str, Any]]:
        """GET /v1/models — list all models in OpenAI format."""
        return self._get("/v1/models").get("data", [])

    # ── Convenience ───────────────────────────────────────────────────────────

    def image(
        self,
        model: str,
        prompt: str,
        output_file: str,
        *,
        steps: int = 20,
        silent: bool = False,
    ) -> str:
        """Generate an image and save it. Returns the output path."""
        data = self.generate_image(model, prompt, output_file, steps=steps, silent=silent)
        if data and output_file:
            with open(output_file, "wb") as f:
                f.write(data)
        return output_file

    def transcribe(
        self,
        audio_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio via /v1/audio/transcriptions (Whisper-compat).
        Requires multipart upload — uses subprocess curl if available, else raises.
        """
        import urllib.parse
        import io

        boundary = "----VortElioFormBoundary7MA4YWxkTrZu0gW"
        body_parts: List[bytes] = []

        with open(audio_path, "rb") as f:
            audio_data = f.read()

        filename = os.path.basename(audio_path)
        body_parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                f"Content-Type: audio/wav\r\n\r\n"
            ).encode()
            + audio_data
            + b"\r\n"
        )
        if model:
            body_parts.append(
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="model"\r\n\r\n'
                    f"{model}\r\n"
                ).encode()
            )
        if language:
            body_parts.append(
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="language"\r\n\r\n'
                    f"{language}\r\n"
                ).encode()
            )
        body_parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(body_parts)

        import urllib.request
        req = urllib.request.Request(
            self._url("/v1/audio/transcriptions"),
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            result = json.loads(resp.read())
        return result.get("text", "")

    def __repr__(self) -> str:
        return f"Vortelio(base={self._base!r})"
