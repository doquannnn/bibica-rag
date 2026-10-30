"""Shared client for the local Qwen VLM served by llama.cpp (`llama-server`, OpenAI-compatible API).

Configure with environment variables:
- BIBICA_LLM_BASE_URL  (default http://127.0.0.1:8080/v1)
- BIBICA_LLM_MODEL     (default "qwen"; llama-server ignores it when only one model is loaded)
- BIBICA_LLM_API_KEY   (default "sk-local"; llama-server only checks it when started with --api-key)
"""
import os
from functools import lru_cache

from openai import OpenAI

BASE_URL = os.environ.get("BIBICA_LLM_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("BIBICA_LLM_MODEL", "qwen")
API_KEY = os.environ.get("BIBICA_LLM_API_KEY", "sk-local")
TIMEOUT = float(os.environ.get("BIBICA_LLM_TIMEOUT", "600"))

# Qwen3.x chat templates think by default; extraction and RAG answers do not need it.
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT)


def server_status() -> str:
    try:
        models = get_client().models.list()
        names = ", ".join(m.id for m in models.data) or "?"
        return f"✅ LLM server {BASE_URL} — model: {names}"
    except Exception as e:  # noqa: BLE001 — shown in the UI, never fatal
        return f"⚠️ Không kết nối được LLM server {BASE_URL}: {e.__class__.__name__}"
