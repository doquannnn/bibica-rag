"""RAPTOR (Sarthi et al. 2024): recursively cluster chunks and summarize each cluster with the LLM, then retrieve over
the collapsed tree (leaves + all summaries) with the usual retriever.

Summaries are cached in data/cache/raptor_<chunking>_<embedder>.json (built on first use).
Retrieval metrics only count LEAF chunks, so broad summaries do not inflate node recall.
"""
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..corpus import CACHE
from ..eval.retrieval_metrics import ranked_nodes
from ..llm import ask, bind
from ..retrieval import DenseEmbedder, build_retriever
from .base import NaiveRAG, _chunks, _lock

SUMMARIZE = """Dưới đây là một nhóm trích đoạn từ các tài liệu nhân sự của Công ty Bibica (QCL = Quy chế tiền lương tiền thưởng, NQLD = Nội quy lao động, TULD = Thỏa ước lao động tập thể):

{texts}

Viết một bản tóm tắt tiếng Việt (150-300 từ) bao quát các quy định chính trong nhóm trích đoạn trên: giữ nguyên các con số, tỷ lệ, thời hạn quan trọng; ghi nguồn dạng [QCL Điều 12]; nếu các tài liệu quy định khác nhau thì nêu rõ. Chỉ in ra bản tóm tắt."""


def kmeans(X, k, iters=50, seed=0):
    rng = np.random.RandomState(seed)
    centers = [X[rng.randint(len(X))]]
    for _ in range(1, k):  # k-means++ init
        d = np.min([1 - X @ c for c in centers], axis=0).clip(min=0)
        centers.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.randint(len(X))])
    C = np.stack(centers)
    for _ in range(iters):
        lab = np.argmax(X @ C.T, axis=1)
        newC = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        newC /= np.linalg.norm(newC, axis=1, keepdims=True)
        if np.allclose(newC, C):
            break
        C = newC
    return lab


def build_tree(chunking, embedder, leaf_per_cluster=6, workers=4):
    path = CACHE / f"raptor_{chunking}_{embedder}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    leaves = _chunks(chunking)
    emb = DenseEmbedder(embedder)
    level_items = [{"text": c["text"], "node_ids": c["node_ids"]} for c in leaves]
    summaries = []
    for level in (1, 2):
        X = emb.embed_docs([it["text"] for it in level_items])
        k = max(2, math.ceil(len(level_items) / leaf_per_cluster))
        lab = kmeans(X, k)
        groups = [[level_items[i] for i in np.where(lab == j)[0]] for j in range(k)]
        groups = [g for g in groups if g]

        def summarize(g):
            texts = "\n\n".join(f"[{', '.join(it['node_ids'])}]\n{it['text'][:3000]}" for it in g)[:24000]
            s = ask(SUMMARIZE.format(texts=texts), max_tokens=700)["text"].strip()
            return {"text": s, "node_ids": list(dict.fromkeys(n for it in g for n in it["node_ids"])),
                    "level": level}

        with ThreadPoolExecutor(workers) as ex:
            level_items = list(ex.map(bind(summarize), groups))
        summaries += level_items
        print(f"raptor level {level}: {len(level_items)} summaries", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summaries, ensure_ascii=False, indent=1), encoding="utf-8")
    return summaries


class RaptorRAG(NaiveRAG):
    def __init__(self, chunking, retriever, top_k=5, embedder="vn-embedding"):
        self.chunking, self.retriever_spec, self.top_k = chunking, retriever, top_k
        leaves = _chunks(chunking)
        summ = build_tree(chunking, embedder)
        self.n_leaves = len(leaves)
        self.chunks = list(leaves) + [
            {"chunk_id": f"raptor-L{s['level']}-{i}", "text": s["text"], "context": f"[Tóm tắt] {s['text']}",
             "node_ids": s["node_ids"], "doc": "MULTI"} for i, s in enumerate(summ)]
        with _lock:
            self.retr = build_retriever(retriever, self.chunks)
        self.name = f"raptor[{chunking}|{retriever}|k{top_k}]"

    def describe(self):
        return {**super().describe(), "name": self.name, "technique": "RAPTOR (collapsed tree)",
                "n_summaries": len(self.chunks) - self.n_leaves}

    def answer(self, question):
        t0 = time.time()
        rk = self.search([question])[0]
        ctx = self.select_contexts(rk)
        out = self.generate(question, ctx, time.time() - t0)
        leaf_rk = [(i, s) for i, s in rk if i < self.n_leaves]
        out["retrieved_nodes"] = ranked_nodes(leaf_rk, self.chunks)[:10]
        return out
