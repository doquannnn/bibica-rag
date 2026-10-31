"""OpenAI-compatible client for the local llama-server (Qwen3.5 VLM), with timing.

The default model/endpoint come from LLM_MODEL / LLM_BASE_URL (benchmark runs use these). Several local models can be
registered for the UI with LLM_ENDPOINTS="qwen3.5-4B=http://127.0.0.1:8080/v1,qwen3.5-9B=http://127.0.0.1:8082/v1"
(default: those two) and selected per request with `use_model(name)`; pipelines built inside `use_model` keep that
model.
"""
import base64
import contextlib
import contextvars
import hashlib
import json
import mimetypes
import os
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen3.5-4B")
ANSWER_MAX_TOKENS = 400  # final answers; prompts ask for concise answers, truncations are counted in the report
RESP_CACHE = Path(__file__).resolve().parents[1] / "data" / "cache" / "llm"


def _endpoints():
    raw = os.environ.get("LLM_ENDPOINTS", "qwen3.5-4B=http://127.0.0.1:8080/v1,qwen3.5-9B=http://127.0.0.1:8082/v1")
    eps = dict(item.strip().split("=", 1) for item in raw.split(",") if "=" in item)
    eps[MODEL] = BASE_URL  # the configured default always points where LLM_BASE_URL says
    return eps


ENDPOINTS = _endpoints()
_current = contextvars.ContextVar("llm_current", default=(MODEL, BASE_URL))


def current():
    """(model, base_url) used by chat() and by pipelines built now."""
    return _current.get()


@contextlib.contextmanager
def use_model(model):
    if model not in ENDPOINTS:
        raise ValueError(f"unknown model {model!r}; known: {', '.join(ENDPOINTS)} (LLM_ENDPOINTS)")
    tok = _current.set((model, ENDPOINTS[model]))
    try:
        yield
    finally:
        _current.reset(tok)


def bind(fn):
    """fn running with the caller's current model, also in worker threads (they do not inherit contextvars)."""
    cur = current()

    def run(*a, **kw):
        tok = _current.set(cur)
        try:
            return fn(*a, **kw)
        finally:
            _current.reset(tok)
    return run


def server_status(model, timeout=2.0):
    """'online' if the endpoint answers and serves this model name, else a short reason."""
    url = ENDPOINTS[model].rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            ids = [m.get("id") for m in json.loads(r.read().decode()).get("data", [])]
    except Exception:
        return "offline"
    return "online" if model in ids else f"server khác model ({', '.join(map(str, ids)) or '?'})"


@lru_cache(maxsize=None)
def _client_for(base_url):
    return OpenAI(base_url=base_url, api_key="local", timeout=900)


def client():
    return _client_for(current()[1])


def image_part(path):
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def chat(messages, max_tokens=ANSWER_MAX_TOKENS, temperature=0.0, json_mode=False, model=None, thinking=False):
    """Returns dict(text, ttft_s, total_s, prompt_tokens, completion_tokens, finish_reason). Streams to measure TTFT.

    Greedy (temperature 0) responses are cached on disk by (model, messages, params); a cache hit returns the
    timings measured when the response was generated, flagged with cached=True."""
    model = model or current()[0]
    key = None
    if temperature == 0:
        key = hashlib.sha1(json.dumps([model, messages, max_tokens, json_mode, thinking], ensure_ascii=False)
                           .encode()).hexdigest()
        path = RESP_CACHE / key[:2] / f"{key}.json"
        if path.exists() and not os.environ.get("LLM_NO_CACHE_READ"):  # set for sequential latency runs
            return {**json.loads(path.read_text(encoding="utf-8")), "cached": True}
    extra = {"chat_template_kwargs": {"enable_thinking": thinking}}
    if json_mode:
        extra["response_format"] = {"type": "json_object"}
    t0 = time.time()
    ttft, parts, usage, finish = None, [], None, None
    stream = client().chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
        top_p=1.0, stream=True, stream_options={"include_usage": True}, extra_body=extra)
    for ev in stream:
        if ev.usage:
            usage = ev.usage
        if ev.choices and ev.choices[0].finish_reason:
            finish = ev.choices[0].finish_reason
        if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
            if ttft is None:
                ttft = time.time() - t0
            parts.append(ev.choices[0].delta.content)
    out = {"text": "".join(parts).strip(), "ttft_s": ttft or (time.time() - t0), "total_s": time.time() - t0,
           "prompt_tokens": usage.prompt_tokens if usage else None,
           "completion_tokens": usage.completion_tokens if usage else None, "finish_reason": finish}
    if key:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def ask(prompt, system=None, **kw):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return chat(msgs, **kw)


# Helpers shared with the PO extraction module (bibica_rag/po).
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def get_client():
    return client()
