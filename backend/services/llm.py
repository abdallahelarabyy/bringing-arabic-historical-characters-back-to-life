"""Chat model clients, called over plain HTTP (no vendor SDK needed).

Providers (LLM_PROVIDER):
  groq    - free API at https://console.groq.com (default when GROQ_API_KEY is set)
  gemini  - Google Gemini API (needs a key whose project has the free tier, or billing)
  openai  - any OpenAI-compatible server, e.g. a local Ollama (http://localhost:11434/v1)

Each client takes a list of {"role": "system" | "user" | "assistant", "content": str} messages
and returns the answer text. When a model is rate-limited or unavailable, the next model in the
list is tried, so the free quotas of several models add up.
"""
import logging
from typing import Dict, List, Optional, Sequence

import requests

log = logging.getLogger("llm")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY_NEXT_MODEL = {404, 408, 413, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class ChatClient:
    provider = "none"

    def __init__(self, models: Sequence[str], temperature: float = 0.4, max_tokens: int = 700,
                 timeout: int = 60):
        self.models = [m for m in models if m]
        if not self.models:
            raise ValueError("no model configured")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    @property
    def model(self) -> str:
        return self.models[0]

    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        last: Optional[LLMError] = None
        for model in self.models:
            try:
                return self._chat(model, messages, self.temperature if temperature is None else temperature)
            except LLMError as e:
                last = e
                if e.status in RETRY_NEXT_MODEL and model != self.models[-1]:
                    log.warning("%s/%s unavailable (%s), trying the next model", self.provider, model, e)
                    continue
                raise
        raise last or LLMError("no model available")

    def _chat(self, model: str, messages: List[Dict[str, str]], temperature: float) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(ChatClient):
    """Groq, Ollama, OpenRouter, ... (POST {base_url}/chat/completions)."""

    def __init__(self, base_url: str, api_key: Optional[str], models: Sequence[str],
                 provider: str = "openai", **kw):
        super().__init__(models, **kw)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.provider = provider

    def _chat(self, model: str, messages: List[Dict[str, str]], temperature: float) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {"model": model, "messages": messages,
                "temperature": temperature, "max_tokens": self.max_tokens}
        if self.provider == "groq":
            # Reasoning models: short spoken answers need no long hidden reasoning (faster, fewer tokens)
            if "qwen3" in model:
                body["reasoning_effort"] = "none"
            elif "gpt-oss" in model:
                body["reasoning_effort"] = "low"
        try:
            r = requests.post(f"{self.base_url}/chat/completions", headers=headers, timeout=self.timeout,
                              json=body)
        except requests.RequestException as e:
            raise LLMError(f"connection failed: {e}", 503) from e
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {_error_text(r)}", r.status_code)
        try:
            return r.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"unexpected response: {r.text[:200]}", 502) from e


class GeminiClient(ChatClient):
    provider = "gemini"

    def __init__(self, api_key: str, models: Sequence[str], **kw):
        super().__init__(models, **kw)
        self.api_key = api_key

    def _chat(self, model: str, messages: List[Dict[str, str]], temperature: float) -> str:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in messages if m["role"] != "system"]
        body = {"contents": contents,
                "generationConfig": {"temperature": temperature, "maxOutputTokens": self.max_tokens}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        try:
            r = requests.post(GEMINI_URL.format(model=model), json=body, timeout=self.timeout,
                              headers={"x-goog-api-key": self.api_key})
        except requests.RequestException as e:
            raise LLMError(f"connection failed: {e}", 503) from e
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {_error_text(r)}", r.status_code)
        try:
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"unexpected response: {r.text[:200]}", 502) from e


def _error_text(r: requests.Response) -> str:
    try:
        err = r.json().get("error")
        if isinstance(err, dict):
            return str(err.get("message", err))[:300]
        return str(err or r.text)[:300]
    except ValueError:
        return r.text[:300]


def create_client(provider: str, *, groq_key: Optional[str], groq_models: Sequence[str],
                  gemini_key: Optional[str], gemini_models: Sequence[str],
                  openai_base_url: str, openai_key: Optional[str], openai_models: Sequence[str]
                  ) -> Optional[ChatClient]:
    """Build the client for LLM_PROVIDER ("auto" picks the first provider that has a key)."""
    provider = (provider or "auto").lower()
    if provider == "auto":
        provider = "groq" if groq_key else "gemini" if gemini_key else "openai" if openai_base_url else "none"
    if provider == "groq" and groq_key:
        return OpenAICompatibleClient(GROQ_BASE_URL, groq_key, groq_models, provider="groq")
    if provider == "gemini" and gemini_key:
        return GeminiClient(gemini_key, gemini_models)
    if provider == "openai" and openai_base_url:
        return OpenAICompatibleClient(openai_base_url, openai_key, openai_models, provider="openai")
    return None
