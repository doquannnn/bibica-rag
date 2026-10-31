"""VLM backend for PO extraction: any OpenAI-compatible chat endpoint that accepts images.

Presets (env `BIBICA_PO_PROVIDER`, default `local`: the Qwen3.5-4B VLM served by `scripts/serve.sh 4B`):

| provider | endpoint | model (default) | key |
|---|---|---|---|
| local    | LLM_BASE_URL (scripts/serve.sh 4B, default http://127.0.0.1:8080/v1) | LLM_MODEL (qwen3.5-4B) | none |
| gemini   | https://generativelanguage.googleapis.com/v1beta/openai/ | gemini-3.5-flash | GEMINI_API_KEY (free tier at aistudio.google.com/apikey) |
| openai   | BIBICA_PO_BASE_URL | BIBICA_PO_MODEL | BIBICA_PO_API_KEY |

`BIBICA_PO_MODEL` / `BIBICA_PO_BASE_URL` / `BIBICA_PO_API_KEY` override any preset.
"""
import os
import time
from dataclasses import dataclass
from functools import lru_cache

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass(frozen=True)
class Backend:
    provider: str
    base_url: str
    model: str
    api_key: str

    @property
    def local(self) -> bool:
        return self.provider == "local"

    def describe(self) -> str:
        return f"{self.provider} · {self.model}"


def backend_from_env(provider: str | None = None, local_model: str | None = None) -> Backend:
    """`local_model`: one of bibica_rag.llm.ENDPOINTS (the UI's model switch) instead of LLM_MODEL / LLM_BASE_URL."""
    env = os.environ
    provider = (provider or env.get("BIBICA_PO_PROVIDER") or "local").lower()
    if provider == "gemini":
        base, model = GEMINI_BASE_URL, "gemini-3.5-flash"
        key = env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or ""
    elif provider == "local":
        base = env.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
        model, key = env.get("LLM_MODEL", "qwen3.5-4B"), "local"
        if local_model:
            from bibica_rag.llm import ENDPOINTS
            base, model = ENDPOINTS[local_model], local_model
    else:
        base, model, key = "", "", ""
    return Backend(provider=provider, base_url=env.get("BIBICA_PO_BASE_URL", base),
                   model=env.get("BIBICA_PO_MODEL", model), api_key=env.get("BIBICA_PO_API_KEY", key))


@lru_cache(maxsize=8)
def _client(base_url: str, api_key: str):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key or "none", timeout=600, max_retries=0)


def get_client(backend: Backend):
    if not backend.base_url:
        raise RuntimeError(f"Provider '{backend.provider}': set BIBICA_PO_BASE_URL")
    if backend.provider == "gemini" and not backend.api_key:
        raise RuntimeError("Thiếu GEMINI_API_KEY (lấy miễn phí tại https://aistudio.google.com/apikey)")
    return _client(backend.base_url, backend.api_key)


def call_with_retry(fn, retries: int = 5, base_delay: float = 4.0):
    """Retry rate limits (free tiers: a few requests per minute) and transient server errors."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            status = getattr(e, "status_code", None)
            transient = status in (429, 500, 502, 503, 504) or e.__class__.__name__ in (
                "APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError")
            msg = str(e).lower()
            if not transient or attempt == retries or "per day" in msg or "perday" in msg:
                raise  # a daily quota does not come back within the retry window: retrying only burns requests
            time.sleep(base_delay * 2 ** attempt)
