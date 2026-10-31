"""Model switch (bibica_rag.llm.use_model / ENDPOINTS) used by the UI: no real model server needed."""
import http.server
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from bibica_rag import llm
from bibica_rag.po.vlm import backend_from_env


def test_endpoints_have_default_and_9b():
    assert llm.ENDPOINTS[llm.MODEL] == llm.BASE_URL
    assert "qwen3.5-9B" in llm.ENDPOINTS


def test_use_model_switches_and_restores():
    assert llm.current() == (llm.MODEL, llm.BASE_URL)
    with llm.use_model("qwen3.5-9B"):
        assert llm.current() == ("qwen3.5-9B", llm.ENDPOINTS["qwen3.5-9B"])
        assert str(llm.client().base_url).rstrip("/") == llm.ENDPOINTS["qwen3.5-9B"].rstrip("/")
    assert llm.current() == (llm.MODEL, llm.BASE_URL)
    with pytest.raises(ValueError):
        with llm.use_model("no-such-model"):
            pass


def test_bind_carries_model_into_worker_threads():
    with llm.use_model("qwen3.5-9B"):
        with ThreadPoolExecutor(2) as ex:
            plain = list(ex.map(lambda _: llm.current()[0], range(2)))
            bound = list(ex.map(llm.bind(lambda _: llm.current()[0]), range(2)))
    assert bound == ["qwen3.5-9B"] * 2
    assert plain == [llm.MODEL] * 2  # why bind() is needed


def _serve(model_ids):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": [{"id": m} for m in model_ids]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_server_status(monkeypatch):
    srv = _serve(["qwen3.5-9B"])
    try:
        url = f"http://127.0.0.1:{srv.server_port}/v1"
        monkeypatch.setitem(llm.ENDPOINTS, "qwen3.5-9B", url)
        monkeypatch.setitem(llm.ENDPOINTS, "qwen3.5-4B", url)
        assert llm.server_status("qwen3.5-9B") == "online"
        assert llm.server_status("qwen3.5-4B").startswith("server khác model")  # 9B is served there, not 4B
    finally:
        srv.shutdown()
    monkeypatch.setitem(llm.ENDPOINTS, "qwen3.5-9B", "http://127.0.0.1:9/v1")
    assert llm.server_status("qwen3.5-9B", timeout=0.5) == "offline"


def test_po_local_backend_follows_selected_model(monkeypatch):
    for k in ("BIBICA_PO_PROVIDER", "BIBICA_PO_MODEL", "BIBICA_PO_BASE_URL", "BIBICA_PO_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    b = backend_from_env("local", local_model="qwen3.5-9B")
    assert (b.model, b.base_url, b.local) == ("qwen3.5-9B", llm.ENDPOINTS["qwen3.5-9B"], True)
    assert backend_from_env("gemini", local_model="qwen3.5-9B").provider == "gemini"
