"""Vortelio asynchronous Python client (requires Python 3.8+ asyncio)."""
from __future__ import annotations

import base64
import json
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

try:
    import aiohttp as _aiohttp_check  # type: ignore
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False


class AsyncVortElioError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(f"HTTP {status}: {message}")


class AsyncConversation:
    """Async stateful multi-turn conversation."""

    def __init__(self, client: "AsyncVortelio", model: str, system: Optional[str] = None) -> None:
        self._client = client
        self._model = model
        self._history: List[Dict[str, Any]] = []
        if system:
            self._history.append({"role": "system", "content": system})

    async def say(self, prompt: str, **kwargs: Any) -> str:
        self._history.append({"role": "user", "content": prompt})
        reply = await self._client.chat(self._model, self._history, **kwargs)
        self._history.append({"role": "assistant", "content": reply})
        return reply

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        self._history.append({"role": "user", "content": prompt})
        parts: List[str] = []
        async for tok in self._client.chat_stream(self._model, self._history, **kwargs):
            parts.append(tok)
            yield tok
        self._history.append({"role": "assistant", "content": "".join(parts)})

    def reset(self, keep_system: bool = True) -> None:
        if keep_system and self._history and self._history[0].get("role") == "system":
            self._history = [self._history[0]]
        else:
            self._history = []


class AsyncVortelio:
    """
    Asynchronous Vortelio client. Requires `aiohttp` (pip install aiohttp).

    Quick start::

        import asyncio
        from vortelio import AsyncVortelio

        async def main():
            ai = AsyncVortelio()
            reply = await ai.chat("llm/mistral:7b", "Hello!")
            print(reply)

            async for tok in ai.chat_stream("llm/mistral:7b", "Tell me a story"):
                print(tok, end="", flush=True)

        asyncio.run(main())
    """

    def __init__(
        self,
        host: str = "http://localhost",
        port: int = 11500,
        timeout: int = 300,
    ) -> None:
        if not _HAS_AIOHTTP:
            raise ImportError("AsyncVortelio requires aiohttp: pip install aiohttp")
        self._base = f"{host.rstrip('/')}:{port}"
        self._timeout = timeout

    def _url(self, path: str) -> str:
        return self._base + path

    async def _get(self, path: str) -> Dict[str, Any]:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self._url(path),
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as resp:
                if resp.status >= 400:
                    body = await resp.json()
                    raise AsyncVortElioError(resp.status, body.get("error", ""))
                return await resp.json()

    async def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._url(path),
                json=body,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as resp:
                if resp.status >= 400:
                    err = await resp.json()
                    raise AsyncVortElioError(resp.status, err.get("error", ""))
                return await resp.json()

    async def _stream_ndjson(
        self,
        path: str,
        body: Dict[str, Any],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._url(path),
                json=body,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as resp:
                async for line in resp.content:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

    async def _stream_sse(
        self,
        path: str,
        body: Dict[str, Any],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._url(path),
                json=body,
                headers={"Accept": "text/event-stream"},
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as resp:
                event_type = "message"
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            return
                        try:
                            obj = json.loads(payload)
                            obj.setdefault("_event", event_type)
                            yield obj
                        except json.JSONDecodeError:
                            pass
                        event_type = "message"

    # ── Server info ───────────────────────────────────────────────────────────

    async def status(self) -> Dict[str, Any]:
        return await self._get("/api/status")

    async def version(self) -> str:
        return (await self._get("/api/version")).get("version", "")

    async def ps(self) -> List[Dict[str, Any]]:
        return (await self._get("/api/ps")).get("models", [])

    # ── Model management ──────────────────────────────────────────────────────

    async def models(self) -> List[Dict[str, Any]]:
        return (await self._get("/api/models")).get("models", [])

    async def tags(self) -> List[Dict[str, Any]]:
        return (await self._get("/api/tags")).get("models", [])

    async def show(self, model: str) -> Dict[str, Any]:
        return await self._post("/api/show", {"model": model})

    async def pull(
        self,
        model: str,
        *,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        silent: bool = False,
    ) -> None:
        async for evt in self._stream_sse("/api/pull", {"model": model, "stream": True}):
            event = evt.get("_event", "message")
            if event == "progress":
                if not silent:
                    pct = evt.get("pct", 0)
                    msg = evt.get("msg", "")
                    print(f"\r  {model}: {pct:3d}%  {msg}   ", end="", flush=True)
                if on_progress:
                    on_progress(evt)
            elif event == "done":
                if not silent:
                    print(f"\r  {model}: done{' ' * 40}")
                return
            elif event == "error":
                raise AsyncVortElioError(500, evt.get("error", "download failed"))

    async def delete(self, model: str) -> None:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                self._url("/api/delete"),
                json={"model": model},
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as resp:
                if resp.status >= 400:
                    raise AsyncVortElioError(resp.status, "delete failed")

    async def copy(self, source: str, destination: str) -> None:
        await self._post("/api/copy", {"source": source, "destination": destination})

    async def create(
        self,
        model: str,
        *,
        modelfile: Optional[str] = None,
        from_model: Optional[str] = None,
        system: Optional[str] = None,
        quantize: Optional[str] = None,
        silent: bool = False,
    ) -> None:
        body: Dict[str, Any] = {"model": model, "stream": True}
        if modelfile:
            body["modelfile"] = modelfile
        if from_model:
            body["from"] = from_model
        if system:
            body["system"] = system
        if quantize:
            body["quantize"] = quantize
        async for chunk in self._stream_ndjson("/api/create", body):
            if not silent:
                print(f"  create: {chunk.get('status', '')}")

    async def quantize(
        self,
        model: str,
        quantize: str,
        *,
        output: Optional[str] = None,
        silent: bool = False,
    ) -> None:
        body: Dict[str, Any] = {"model": model, "quantize": quantize, "stream": True}
        if output:
            body["output"] = output
        async for chunk in self._stream_ndjson("/api/quantize", body):
            if not silent:
                print(f"  quantize: {chunk.get('status', '')}")
            if chunk.get("error"):
                raise AsyncVortElioError(500, chunk["error"])

    # ── LLM — generate ────────────────────────────────────────────────────────

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[List[str]] = None,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        think: bool = False,
        raw: bool = False,
        context: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if system:
            body["system"] = system
        if images:
            body["images"] = images
        if format:
            body["format"] = format
        if options:
            body["options"] = options
        if think:
            body["think"] = True
        if raw:
            body["raw"] = True
        if context:
            body["context"] = context
        return await self._post("/api/generate", body)

    async def generate_stream(
        self,
        model: str,
        prompt: str,
        *,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        think: bool = False,
    ) -> AsyncGenerator[str, None]:
        body: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": True}
        if system:
            body["system"] = system
        if options:
            body["options"] = options
        if think:
            body["think"] = True
        async for chunk in self._stream_ndjson("/api/generate", body):
            if chunk.get("done"):
                return
            tok = chunk.get("response", "")
            if tok:
                yield tok

    # ── LLM — chat ────────────────────────────────────────────────────────────

    async def chat(
        self,
        model: str,
        messages: Union[List[Dict[str, Any]], str],
        *,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        think: bool = False,
    ) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if format:
            body["format"] = format
        if options:
            body["options"] = options
        if tools:
            body["tools"] = tools
        if think:
            body["think"] = True
        result = await self._post("/api/chat", body)
        return result.get("message", {}).get("content", "")

    async def chat_stream(
        self,
        model: str,
        messages: Union[List[Dict[str, Any]], str],
        *,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        think: bool = False,
    ) -> AsyncGenerator[str, None]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if format:
            body["format"] = format
        if options:
            body["options"] = options
        if tools:
            body["tools"] = tools
        if think:
            body["think"] = True
        async for chunk in self._stream_ndjson("/api/chat", body):
            if chunk.get("done"):
                return
            tok = chunk.get("message", {}).get("content", "")
            if tok:
                yield tok

    def conversation(self, model: str, *, system: Optional[str] = None) -> AsyncConversation:
        return AsyncConversation(self, model, system=system)

    # ── Embeddings ────────────────────────────────────────────────────────────

    async def embed(
        self,
        model: str,
        input: Union[str, List[str]],
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[List[float]]:
        body: Dict[str, Any] = {"model": model, "input": input}
        if options:
            body["options"] = options
        return (await self._post("/api/embed", body)).get("embeddings", [])

    async def embeddings(self, model: str, prompt: str) -> List[float]:
        return (await self._post("/api/embeddings", {"model": model, "prompt": prompt})).get("embedding", [])

    # ── Advanced ──────────────────────────────────────────────────────────────

    async def route(self, task: str, **kwargs: Any) -> Dict[str, Any]:
        return await self._post("/api/route", {"task": task, **kwargs})

    async def compare(self, models: List[str], prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return await self._post("/api/compare", {"models": models, "prompt": prompt, **kwargs})

    async def structured(
        self, model: str, prompt: str, schema: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model, "prompt": prompt, **kwargs}
        if schema:
            body["schema"] = schema
        return await self._post("/api/structured", body)

    async def summarize(self, model: str, text: str, **kwargs: Any) -> Dict[str, Any]:
        return await self._post("/api/summarize", {"model": model, "text": text, **kwargs})

    async def think(self, model: str, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return await self._post("/api/think", {"model": model, "prompt": prompt, **kwargs})

    # ── RAG ───────────────────────────────────────────────────────────────────

    async def rag_ingest(
        self,
        model: str,
        documents: List[Dict[str, Any]],
        *,
        collection: str = "default",
        chunk_size: int = 800,
    ) -> Dict[str, Any]:
        return await self._post("/api/rag/ingest", {
            "model": model, "collection": collection,
            "documents": documents, "chunk_size": chunk_size,
        })

    async def rag_query(
        self, model: str, query: str, *, collection: str = "default", top_k: int = 5
    ) -> Dict[str, Any]:
        return await self._post("/api/rag/query", {
            "model": model, "collection": collection, "query": query, "top_k": top_k,
        })

    # ── Agents ────────────────────────────────────────────────────────────────

    async def agents_catalog(self) -> List[Dict[str, Any]]:
        return (await self._get("/api/agents/catalog")).get("agents", [])

    async def agents_install(self, agent_id: str) -> Dict[str, Any]:
        return await self._post("/api/agents/install", {"agent": agent_id})

    async def agents_start(self, agent_id: str) -> Dict[str, Any]:
        return await self._post("/api/agents/start", {"agent": agent_id})

    async def agents_stop(self, agent_id: str) -> Dict[str, Any]:
        return await self._post("/api/agents/stop", {"agent": agent_id})

    # ── OpenAI-compat ─────────────────────────────────────────────────────────

    async def openai_chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if tools:
            body["tools"] = tools
        return await self._post("/v1/chat/completions", body)

    async def openai_chat_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        body: Dict[str, Any] = {
            "model": model, "messages": messages, "stream": True,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        async for evt in self._stream_sse("/v1/chat/completions", body):
            for choice in evt.get("choices", []):
                tok = choice.get("delta", {}).get("content", "")
                if tok:
                    yield tok

    async def openai_embeddings(self, model: str, input: Union[str, List[str]]) -> Dict[str, Any]:
        return await self._post("/v1/embeddings", {"model": model, "input": input})

    async def openai_models(self) -> List[Dict[str, Any]]:
        return (await self._get("/v1/models")).get("data", [])

    def __repr__(self) -> str:
        return f"AsyncVortelio(base={self._base!r})"
