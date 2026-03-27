"""
Vortelio + FastAPI — async streaming chat endpoint.
  pip install vortelio fastapi uvicorn
  vortelio serve
  uvicorn fastapi_streaming:app --reload
  curl "http://localhost:8000/chat?q=Hello"
"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from vortelio import Vortelio
import json

app = FastAPI(title="Vortelio Chat API")
ai  = Vortelio()

@app.get("/chat")
def chat(q: str = "Hello!", model: str = "llm/mistral:7b"):
    """Stream tokens from a local AI model."""
    def generate():
        for token in ai.stream(model, q):
            yield f"data: {json.dumps(token)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/models")
def models():
    """List installed models."""
    return ai._get("/api/models")

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <script>
    function ask() {
        const q = document.getElementById('q').value;
        const out = document.getElementById('out');
        out.textContent = '';
        const es = new EventSource('/chat?q=' + encodeURIComponent(q));
        es.onmessage = e => {
            if (e.data === '[DONE]') { es.close(); return; }
            out.textContent += JSON.parse(e.data);
        };
    }
    </script>
    <h2>Vortelio Chat</h2>
    <input id="q" value="What is Python?" size="60">
    <button onclick="ask()">Ask</button>
    <pre id="out" style="border:1px solid #ccc;padding:1em;min-height:100px"></pre>
    """
