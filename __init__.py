"""
PullAI + Flask — streaming chat endpoint.
  pip install pullai flask
  pullai serve
  python flask_streaming.py
  curl "http://localhost:5000/chat?q=Hello"
"""
from flask import Flask, Response, stream_with_context, request
from pullai import PullAI
import json

app = Flask(__name__)
ai  = PullAI()

@app.route("/chat")
def chat():
    prompt = request.args.get("q", "Hello!")
    model  = request.args.get("model", "llm/mistral:7b")

    def generate():
        for token in ai.stream(model, prompt):
            yield f"data: {json.dumps(token)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.route("/")
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
    <input id="q" value="What is Python?" size="50">
    <button onclick="ask()">Ask</button>
    <pre id="out"></pre>
    """

if __name__ == "__main__":
    app.run(debug=True)
