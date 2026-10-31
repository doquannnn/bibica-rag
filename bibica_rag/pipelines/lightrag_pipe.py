"""LightRAG (graph RAG): the local Qwen extracts an entity/relation graph from every node at index time; at query time
LightRAG's "mix" mode (KG local+global + vector chunks) builds the context, which is then answered with the SAME
QA prompt as every other pipeline (only_need_context=True), so only the retrieval side differs.

Build once:  uv run python -m bibica_rag.pipelines.lightrag_pipe build
"""
import asyncio
import json
import re
import sys
import threading
import time

import numpy as np

from ..corpus import CACHE, load_nodes
from ..llm import BASE_URL, MODEL
from ..retrieval import DenseEmbedder
from .base import NaiveRAG, Pipeline, _lock

NODE_ID = re.compile(r"\[((?:QCL|NQLD|TULD)/(?:Điều \d+|Phụ lục \d+|Chương [IVX]+|Mở đầu))\]")


def _make_rag(embedder):
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.utils import EmbeddingFunc

    emb = DenseEmbedder(embedder)

    async def llm_func(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kw):
        kw.pop("hashing_kv", None)
        kw.setdefault("max_tokens", 3000)  # big tables can make the 4B model loop until the client times out
        kw.setdefault("timeout", 1500)
        return await openai_complete_if_cache(
            MODEL, prompt, system_prompt=system_prompt, history_messages=history_messages or [],
            base_url=BASE_URL, api_key="local", keyword_extraction=keyword_extraction,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}, temperature=0, **kw)

    async def embed_func(texts):
        with _lock:
            return np.asarray(emb.embed_docs(list(texts)))

    rag = LightRAG(working_dir=str(CACHE / f"lightrag_{embedder}"), llm_model_func=llm_func, llm_model_name=MODEL,
                   embedding_func=EmbeddingFunc(embedding_dim=1024, max_token_size=1024, func=embed_func),
                   addon_params={"language": "Vietnamese"}, llm_model_max_async=8, chunk_token_size=1200,
                   entity_extract_max_gleaning=0, max_parallel_insert=4, enable_llm_cache=True,
                   default_llm_timeout=1500)  # long tables (Phụ lục 2) time out at the 480 s default

    async def init():
        await rag.initialize_storages()
        await initialize_pipeline_status()

    return rag, init


class LightRAGPipe(Pipeline):
    framework = "lightrag"

    def __init__(self, embedder="vn-embedding", mode="mix", top_k=5, max_tokens=6000):
        from lightrag import QueryParam

        self.mode, self.top_k = mode, top_k
        self.rag, init = _make_rag(embedder)
        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete(init())
        self.param = QueryParam(mode=mode, only_need_context=True, chunk_top_k=top_k, top_k=20,
                                max_total_tokens=max_tokens, enable_rerank=False)
        self.loop_lock = threading.Lock()
        self.name = f"lightrag[{embedder}|{mode}|k{top_k}]"
        self.embedder = embedder

    def describe(self):
        return {"name": self.name, "framework": self.framework, "chunking": "per-node (LightRAG chunker)",
                "retriever": f"LightRAG {self.mode} (KG + dense:{self.embedder})", "top_k": self.top_k,
                "technique": "graph RAG (LightRAG)"}

    def answer(self, question):
        t0 = time.time()
        with self.loop_lock:  # one asyncio loop shared by worker threads
            ctx_text = self.loop.run_until_complete(self.rag.aquery(question, param=self.param)) or ""
        nodes = list(dict.fromkeys(NODE_ID.findall(ctx_text)))
        ctx = [{"node_ids": nodes, "context": ctx_text}]
        try:
            out = NaiveRAG.generate(self, question, ctx, time.time() - t0)
        except Exception as e:  # the KG context can exceed a serving slot (8k tokens): truncate and retry once
            if "exceed_context_size" not in str(e):
                raise
            ctx = [{"node_ids": nodes, "context": ctx_text[:12000]}]
            out = NaiveRAG.generate(self, question, ctx, time.time() - t0)
            out["context_truncated"] = True
        out.update(retrieved_nodes=nodes[:10], llm_calls=2)
        return out


def build(embedder="vn-embedding"):
    rag, init = _make_rag(embedder)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(init())
    nodes = list(load_nodes())
    texts = [f"[{n['node_id']}] {n['doc_title']} — {n['title']}\n{n['text']}" for n in nodes]
    t0 = time.time()
    # LightRAG de-duplicates documents by file BASENAME, so "QCL/Điều 3" and "NQLD/Điều 3" would collide
    paths = [n["node_id"].replace("/", "_").replace(" ", "_") + ".md" for n in nodes]
    loop.run_until_complete(rag.ainsert(texts, ids=[n["node_id"] for n in nodes], file_paths=paths))
    print(f"LightRAG index built in {(time.time() - t0) / 60:.1f} min", flush=True)


def retry_failed(embedder="vn-embedding"):
    """Re-run extraction for documents left FAILED (e.g. LLM timeouts) by a previous build."""
    rag, init = _make_rag(embedder)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(init())
    status = json.loads((CACHE / f"lightrag_{embedder}" / "kv_store_doc_status.json").read_text(encoding="utf-8"))
    failed = [k for k, v in status.items() if v.get("status") == "failed"]
    by_id = {n["node_id"]: n for n in load_nodes()}
    for doc_id in failed:  # failed docs are not re-picked by the queue: delete, then insert again
        loop.run_until_complete(rag.adelete_by_doc_id(doc_id))
    nodes = [by_id[d] for d in failed]
    loop.run_until_complete(rag.ainsert(
        [f"[{n['node_id']}] {n['doc_title']} — {n['title']}\n{n['text']}" for n in nodes],
        ids=[n["node_id"] for n in nodes],
        file_paths=[n["node_id"].replace("/", "_").replace(" ", "_") + ".md" for n in nodes]))
    print("retried", failed, flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build(*(sys.argv[2:3] or []))
    elif sys.argv[1:2] == ["retry"]:
        retry_failed(*(sys.argv[2:3] or []))
