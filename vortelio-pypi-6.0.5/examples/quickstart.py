"""
Vortelio SDK — Quick Start
  pip install vortelio
  vortelio serve      (in a separate terminal)
  python quickstart.py
"""
from vortelio import Vortelio

ai = Vortelio()

# ── Status ────────────────────────────────────────────────────────────────
info = ai.status()
print(f"Vortelio {info['version']} — {info['hardware']}")

# ── Models ────────────────────────────────────────────────────────────────
ai.models()

# ── Chat (default: tokens printed in real time) ───────────────────────────
reply = ai.chat("llm/qwen:0.5b", "What is Python in one sentence?")
print()

# ── on_token callback ─────────────────────────────────────────────────────
print("Custom callback:")
tokens = []
ai.chat("llm/qwen:0.5b", "Say hello",
        on_token=lambda t: tokens.append(t),
        silent=True)
print("".join(tokens))

# ── stream() generator ────────────────────────────────────────────────────
print("\nGenerator:")
for token in ai.stream("llm/qwen:0.5b", "Count to 5"):
    print(token, end="", flush=True)
print()

# ── Conversation with token streaming ─────────────────────────────────────
conv = ai.conversation("llm/qwen:0.5b", system="You are a helpful assistant.")
conv.say("My name is Marco.")
for token in conv.stream("What is my name?"):
    print(token, end="", flush=True)
print()
