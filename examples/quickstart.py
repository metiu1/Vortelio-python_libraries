"""
Vortelio Python SDK — Quick Start
  pip install vortelio
  vortelio serve      (in another terminal)
  python examples/quickstart.py
"""
from vortelio import Vortelio

ai = Vortelio()

# ── Server info ───────────────────────────────────────────────────────────────
print("Server version:", ai.version())

# ── Models ────────────────────────────────────────────────────────────────────
models = ai.models()
print(f"Downloaded models: {len(models)}")
for m in models[:5]:
    print(f"  {m.get('model', {}).get('type','?')}/{m.get('model', {}).get('name','?')}")

# ── Chat (streams to stdout) ──────────────────────────────────────────────────
MODEL = "llm/qwen:0.5b"   # change to any model you have downloaded

print("\nChat (streaming):")
reply = ai.chat(MODEL, "What is Python in one sentence?")
print()

# ── Generator stream ─────────────────────────────────────────────────────────
print("\nGenerator stream:")
for tok in ai.chat_stream(MODEL, "Count from 1 to 5"):
    print(tok, end="", flush=True)
print()

# ── Conversation ──────────────────────────────────────────────────────────────
print("\nConversation:")
conv = ai.conversation(MODEL, system="You are a helpful assistant.")
conv.say("My name is Marco.")
print()
reply = conv.say("What is my name?")
print()

# ── Generate (Ollama-style) ───────────────────────────────────────────────────
result = ai.generate(MODEL, "The capital of France is", options={"num_predict": 10})
print("\nGenerate result:", result["response"])

# ── Structured output ─────────────────────────────────────────────────────────
result = ai.structured(MODEL, "List 3 colors as JSON array of strings")
print("\nStructured:", result.get("parsed") or result.get("raw"))

# ── Think (chain-of-thought) ──────────────────────────────────────────────────
result = ai.think(MODEL, "Is 17 prime?")
print("\nThinking:", result.get("thinking", "")[:100])
print("Answer:", result.get("answer", ""))
