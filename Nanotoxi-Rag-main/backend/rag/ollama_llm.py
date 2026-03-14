from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .config import LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE, OLLAMA_BASE_URL, OLLAMA_MODEL


@dataclass(frozen=True)
class OllamaResult:
    text: str
    raw: Any | None = None


class OllamaLLM:
    """
    Local LLM via Ollama. Requires Ollama running on the host machine.

    Default URL: http://localhost:11434
    Uses /api/chat with stream=false.
    """

    def __init__(
        self,
        *,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s

    def generate(self, prompt: str) -> OllamaResult:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are a scientific assistant. Use only the provided context."},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "num_predict": int(self.max_new_tokens),
                "temperature": float(self.temperature),
            },
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        text = ((data.get("message") or {}).get("content") or "").strip()
        return OllamaResult(text=text, raw=data)

