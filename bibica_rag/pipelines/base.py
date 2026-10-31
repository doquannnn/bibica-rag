"""Pipeline interface + the hand-written pipelines (naive RAG, multi-query / HyDE, full-context CAG).

Every pipeline's answer(question) returns:
    {answer, contexts: [{node_ids, context}], retrieved_nodes, timings: {retrieval_s, ttft_s, total_s}, llm_calls}
"""
import threading
import time
from functools import lru_cache

from .. import prompts
from ..corpus import full_corpus_text
from ..eval.retrieval_metrics import ranked_nodes
from ..llm import ANSWER_MAX_TOKENS, ask, chat
from ..retrieval import rrf

CTX_BUDGET = 6000  # max chars of retrieved context handed to the generator
_lock = threading.Lock()  # torch models (embedders / rerankers) are not thread-safe under concurrent calls


class Pipeline:
    name = "base"
    framework = "python"

    def describe(self) -> dict:
        return {"name": self.name, "framework": self.framework}

    def answer(self, question: str) -> dict:
        raise NotImplementedError


@lru_cache(maxsize=None)
def _chunks(chunking):
    from ..eval.stage_a import build_chunks
    return build_chunks(chunking)


@lru_cache(maxsize=None)
def _retriever(chunking, spec):
    from ..retrieval import build_retriever
    return build_retriever(spec, _chunks(chunking))


class NaiveRAG(Pipeline):
    """Retrieve with a Stage-A configuration, put the top chunks (≤ CTX_BUDGET chars) in the prompt, generate once."""

    framework = "python"

    def __init__(self, chunking, retriever, top_k=5):
        self.chunking, self.retriever_spec, self.top_k = chunking, retriever, top_k
        self.chunks = _chunks(chunking)
        with _lock:
            self.retr = _retriever(chunking, retriever)
        self.name = f"naive[{chunking}|{retriever}|k{top_k}]"

    def describe(self):
        return {"name": self.name, "framework": self.framework, "chunking": self.chunking,
                "retriever": self.retriever_spec, "top_k": self.top_k, "technique": "naive"}

    def search(self, queries, k=20):
        with _lock:
            return self.retr.search(list(queries), k=k)

    def select_contexts(self, ranking):
        """Top-k chunks within the char budget; small-to-big parents are de-duplicated."""
        out, seen, used = [], set(), 0
        for idx, _ in ranking:
            c = self.chunks[idx]
            if c["context"] in seen:
                continue
            if out and used + len(c["context"]) > CTX_BUDGET:
                continue
            seen.add(c["context"])
            out.append({"node_ids": c["node_ids"], "context": c["context"]})
            used += len(c["context"])
            if len(out) >= self.top_k:
                break
        return out

    def generate(self, question, contexts, t_retr, max_tokens=ANSWER_MAX_TOKENS):
        t0 = time.time()
        r = chat(prompts.qa_messages(question, contexts), max_tokens=max_tokens)
        return {"answer": r["text"], "contexts": contexts,
                "timings": {"retrieval_s": t_retr, "ttft_s": t_retr + r["ttft_s"], "total_s": t_retr + time.time() - t0},
                "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                "finish_reason": r.get("finish_reason"), "cached": r.get("cached", False), "llm_calls": 1}

    def answer(self, question):
        t0 = time.time()
        rk = self.search([question])[0]
        ctx = self.select_contexts(rk)
        out = self.generate(question, ctx, time.time() - t0)
        out["retrieved_nodes"] = ranked_nodes(rk, self.chunks)[:10]
        return out


class QueryTransformRAG(NaiveRAG):
    """multiquery: LLM rewrites the question into n search queries (also fixes missing diacritics) — RRF fusion.
    hyde: LLM writes a hypothetical regulation passage; search with question + passage, RRF fusion."""

    def __init__(self, mode, chunking, retriever, top_k=5, n=3):
        super().__init__(chunking, retriever, top_k)
        self.mode, self.n = mode, n
        self.name = f"{mode}[{chunking}|{retriever}|k{top_k}]"

    def describe(self):
        return {**super().describe(), "name": self.name, "technique": self.mode}

    def answer(self, question):
        t0 = time.time()
        if self.mode == "multiquery":
            r = ask(prompts.MULTI_QUERY.format(n=self.n, q=question), max_tokens=120)
            queries = [question] + [l.strip(" -•\t") for l in r["text"].splitlines() if l.strip()][: self.n]
        else:
            r = ask(prompts.HYDE.format(q=question), max_tokens=200)
            queries = [question, r["text"]]
        rankings = self.search(queries)
        rk = rrf(rankings, top=20)
        ctx = self.select_contexts(rk)
        out = self.generate(question, ctx, time.time() - t0)
        out.update(retrieved_nodes=ranked_nodes(rk, self.chunks)[:10], llm_calls=2, queries=queries)
        return out


class FullContextCAG(Pipeline):
    """No retrieval: the whole corpus (~23.5k tokens) is the prompt prefix; llama.cpp reuses its KV cache across
    questions (cache-augmented generation). Needs a server slot with ≥ 28k context."""

    framework = "python"
    name = "cag[full-corpus]"

    def __init__(self):
        self.corpus = full_corpus_text()

    def describe(self):
        return {"name": self.name, "framework": self.framework, "chunking": "none", "retriever": "none",
                "technique": "full-context (CAG)"}

    def answer(self, question):
        t0 = time.time()
        msgs = [{"role": "system", "content": prompts.SYSTEM_QA + "\n\nTOÀN VĂN TÀI LIỆU NỘI BỘ:\n\n" + self.corpus},
                {"role": "user", "content": f"CÂU HỎI: {question}\n\nTRẢ LỜI:"}]
        r = chat(msgs)
        return {"answer": r["text"], "contexts": [], "retrieved_nodes": [],
                "timings": {"retrieval_s": 0.0, "ttft_s": r["ttft_s"], "total_s": time.time() - t0},
                "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"], "llm_calls": 1}
