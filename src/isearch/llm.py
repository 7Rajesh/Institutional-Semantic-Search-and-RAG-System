"""Minimal Ollama client (no extra dependencies)."""
from __future__ import annotations

import re
import time

import requests

from .config import Settings


class OllamaClient:
    def __init__(self, settings: Settings):
        self.url, self.model = settings.ollama_url, settings.ollama_model
        self._checked_at, self._available = 0.0, False

    def available(self, max_age: float = 15.0) -> bool:
        if time.time() - self._checked_at < max_age:
            return self._available
        try:
            tags = requests.get(f"{self.url}/api/tags", timeout=2).json().get("models", [])
            self._available = any(m["name"] == self.model or m["name"].startswith(self.model + ":") for m in tags)
        except Exception:
            self._available = False
        self._checked_at = time.time()
        return self._available

    def chat(self, messages, temperature: float = 0.0, max_tokens: int = 400) -> str:
        resp = requests.post(
            f"{self.url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False,
                  "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 4096}},
            timeout=180,
        )
        resp.raise_for_status()
        text = resp.json()["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()      # strip deepseek-r1 reasoning
