"""OpenAI-compatible client for the local llama-server (Qwen3.5 VLM), with timing."""
import base64
import hashlib
import json
import mimetypes
import os
import time
from pathlib import Path

from openai import OpenAI

BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen3.5-4B")
ANSWER_MAX_TOKENS = 400  # final answers; prompts ask for concise answers, truncations are counted in the report
RESP_CACHE = Path(__file__).resolve().parents[1] / "data" / "cache" / "llm"

_client = None


def client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=BASE_URL, api_key="local", timeout=900)
    return _client


def image_part(path):
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def chat(messages, max_tokens=ANSWER_MAX_TOKENS, temperature=0.0, json_mode=False, model=None, thinking=False):
    """Returns dict(text, ttft_s, total_s, prompt_tokens, completion_tokens, finish_reason). Streams to measure TTFT.

    Greedy (temperature 0) responses are cached on disk by (model, messages, params); a cache hit returns the
    timings measured when the response was generated, flagged with cached=True."""
    model = model or MODEL
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
