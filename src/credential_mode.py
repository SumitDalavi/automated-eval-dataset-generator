import os

def is_live_openai_key(key: str = None) -> bool:
    if key is None:
        key = os.getenv("OPENAI_API_KEY", "")
    return bool(key) and key.startswith("sk-") and not key.startswith("sk-dummy") and len(key) >= 20

def detect_mode():
    return {
        "llm": "live" if is_live_openai_key() else "mock",
        "strategies": ["qa", "adversarial"]
    }
