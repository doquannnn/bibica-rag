"""The same RAG recipe implemented idiomatically in LangChain, LangGraph (agentic CRAG), LlamaIndex, Haystack, DSPy.

All use the same chunks, embedding model, LLM endpoint and system prompt, so differences come from each framework's
retrieval components, orchestration and prompt assembly.
"""
import json
import re
import time

from .. import prompts
from ..chunking import STRATEGIES
from ..eval.retrieval_metrics import ranked_nodes
from ..llm import ANSWER_MAX_TOKENS, API_KEY, BASE_URL, EXTRA_BODY, MODEL, ask
from ..retrieval import DenseEmbedder, get_reranker, tok_fold_plus
from .base import NaiveRAG, Pipeline, _lock


def _timed_result(answer, contexts, t_retr, t0, nodes, calls=1):
    total = time.time() - t0
    return {"answer": answer.strip(), "contexts": contexts, "retrieved_nodes": nodes,
            "timings": {"retrieval_s": t_retr, "ttft_s": total, "total_s": total}, "llm_calls": calls}


class _FrameworkBase(Pipeline):
    def __init__(self, chunking, embedder, reranker=None, top_k=5, depth=20):
        self.chunking, self.embedder_name, self.reranker_name = chunking, embedder, reranker
        self.top_k, self.depth = top_k, depth
        self.chunks = STRATEGIES[chunking]()
        self.emb = DenseEmbedder(embedder)
        self.name = f"{self.framework}[{chunking}|hybrid:{embedder}+bm25" + (f"|rr:{reranker}" if reranker else "") + \
                    f"|k{top_k}]"

    def describe(self):
        return {"name": self.name, "framework": self.framework, "chunking": self.chunking,
                "retriever": f"hybrid:dense:{self.embedder_name}+bm25" + (f"|rerank:{self.reranker_name}"
                                                                           if self.reranker_name else ""),
                "top_k": self.top_k, "technique": "naive"}

    def _rerank(self, q, items, text_of):
        if not self.reranker_name:
            return items[: self.top_k]
        with _lock:
            s = get_reranker(self.reranker_name).predict([(q, text_of(x)) for x in items])
        order = sorted(range(len(items)), key=lambda i: -float(s[i]))
        return [items[i] for i in order[: self.top_k]]


# ---------------------------------------------------------------- LangChain

class LangChainRAG(_FrameworkBase):
    framework = "langchain"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        from langchain_classic.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document
        from langchain_core.embeddings import Embeddings
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.vectorstores import InMemoryVectorStore
        from langchain_openai import ChatOpenAI

        emb = self.emb

        class STEmb(Embeddings):
            def embed_documents(self, texts):
                return emb.embed_docs(texts).tolist()

            def embed_query(self, text):
                return emb.embed_queries([text])[0].tolist()

        docs = [Document(page_content=c["text"], metadata={"node_ids": c["node_ids"], "context": c["context"]})
                for c in self.chunks]
        vs = InMemoryVectorStore.from_documents(docs, STEmb())
        bm25 = BM25Retriever.from_documents(docs, preprocess_func=tok_fold_plus, k=self.depth)
        self.retriever = EnsembleRetriever(retrievers=[vs.as_retriever(search_kwargs={"k": self.depth}), bm25],
                                           weights=[0.5, 0.5], c=60)
        prompt = ChatPromptTemplate.from_messages([("system", prompts.SYSTEM_QA),
                                                   ("human", "TRÍCH ĐOẠN TÀI LIỆU:\n{context}\n\nCÂU HỎI: {question}"
                                                             "\n\nTRẢ LỜI:")])
        llm = ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, model=MODEL, temperature=0, max_tokens=ANSWER_MAX_TOKENS,
                         extra_body=EXTRA_BODY, timeout=900)
        self.chain = prompt | llm | StrOutputParser()

    def answer(self, question):
        t0 = time.time()
        with _lock:
            docs = self.retriever.invoke(question)
        docs = self._rerank(question, docs, lambda d: d.page_content)
        ctx, seen = [], set()
        for d in docs:
            if d.metadata["context"] not in seen:
                seen.add(d.metadata["context"])
                ctx.append({"node_ids": d.metadata["node_ids"], "context": d.metadata["context"]})
        t_retr = time.time() - t0
        ans = self.chain.invoke({"context": prompts.format_contexts(ctx), "question": question})
        nodes = list(dict.fromkeys(n for d in docs for n in d.metadata["node_ids"]))
        return _timed_result(ans, ctx, t_retr, t0, nodes)


# ---------------------------------------------------------------- LangGraph (agentic corrective RAG)

class LangGraphCRAG(Pipeline):
    """retrieve -> grade sufficiency (LLM, JSON) -> [rewrite query -> retrieve again, merge] -> generate."""

    framework = "langgraph"

    def __init__(self, chunking, retriever, top_k=5, max_loops=1):
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        self.base = NaiveRAG(chunking, retriever, top_k)
        self.name = f"langgraph-crag[{chunking}|{retriever}|k{top_k}]"
        self.max_loops = max_loops
        base = self.base

        class S(TypedDict, total=False):
            question: str
            query: str
            ranking: list
            contexts: list
            loops: int
            calls: int
            answer: str
            gen: dict

        def retrieve(s):
            rk = base.search([s["query"]])[0]
            prev = s.get("ranking", [])
            merged = rk if not prev else __import__("bibica_rag.retrieval", fromlist=["rrf"]).rrf([prev, rk], top=20)
            return {"ranking": merged, "contexts": base.select_contexts(merged)}

        def grade(s):
            r = ask(prompts.GRADE_DOCS.format(q=s["question"], ctx=prompts.format_contexts(s["contexts"])),
                    max_tokens=120, json_mode=True)
            try:
                g = json.loads(r["text"])
            except json.JSONDecodeError:
                g = {"sufficient": True}
            upd = {"calls": s.get("calls", 0) + 1, "loops": s.get("loops", 0) + 1}
            if not g.get("sufficient", True) and g.get("better_query"):
                upd["query"] = g["better_query"]
                upd["answer"] = "__retry__"
            else:
                upd["answer"] = ""
            return upd

        def route(s):
            return "retrieve" if s.get("answer") == "__retry__" and s["loops"] <= self.max_loops else "generate"

        def generate(s):
            g = base.generate(s["question"], s["contexts"], 0.0)
            return {"answer": g["answer"], "gen": g, "calls": s.get("calls", 0) + 1}

        g = StateGraph(S)
        g.add_node("retrieve", retrieve)
        g.add_node("grade", grade)
        g.add_node("generate", generate)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", route, {"retrieve": "retrieve", "generate": "generate"})
        g.add_edge("generate", END)
        self.graph = g.compile()

    def describe(self):
        return {**self.base.describe(), "name": self.name, "framework": self.framework,
                "technique": "agentic corrective RAG (grade + rewrite)"}

    def answer(self, question):
        t0 = time.time()
        s = self.graph.invoke({"question": question, "query": question, "loops": 0, "calls": 0})
        total = time.time() - t0
        g = s["gen"]
        return {"answer": s["answer"], "contexts": s["contexts"],
                "retrieved_nodes": ranked_nodes(s["ranking"], self.base.chunks)[:10],
                "timings": {"retrieval_s": total - g["timings"]["total_s"],
                            "ttft_s": total - g["timings"]["total_s"] + g["timings"]["ttft_s"], "total_s": total},
                "llm_calls": s["calls"], "loops": s["loops"]}


# ---------------------------------------------------------------- LlamaIndex

class LlamaIndexRAG(_FrameworkBase):
    framework = "llamaindex"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        from llama_index.core import PromptTemplate, VectorStoreIndex
        from llama_index.core.embeddings import BaseEmbedding
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.core.schema import TextNode
        from llama_index.llms.openai_like import OpenAILike

        emb = self.emb

        class STEmb(BaseEmbedding):
            def _get_query_embedding(self, query):
                return emb.embed_queries([query])[0].tolist()

            async def _aget_query_embedding(self, query):
                return self._get_query_embedding(query)

            def _get_text_embedding(self, text):
                return emb.embed_docs([text])[0].tolist()

            def _get_text_embeddings(self, texts):
                return emb.embed_docs(texts).tolist()

        nodes = [TextNode(text=c["text"], id_=c["chunk_id"],
                          metadata={"node_ids": ",".join(c["node_ids"]), "context": c["context"]},
                          excluded_embed_metadata_keys=["node_ids", "context"],
                          excluded_llm_metadata_keys=["context"]) for c in self.chunks]
        self.llm = OpenAILike(model=MODEL, api_base=BASE_URL, api_key=API_KEY, is_chat_model=True, temperature=0,
                              max_tokens=ANSWER_MAX_TOKENS, timeout=900, additional_kwargs={"extra_body": EXTRA_BODY})
        index = VectorStoreIndex(nodes, embed_model=STEmb(embed_batch_size=8))
        vec = index.as_retriever(similarity_top_k=self.depth)
        bm25 = _LIBM25(nodes, self.depth)
        self.retriever = QueryFusionRetriever([vec, bm25], similarity_top_k=self.depth, num_queries=1,
                                              mode="reciprocal_rerank", use_async=False, llm=self.llm)
        self.qa_tmpl = PromptTemplate(prompts.SYSTEM_QA + "\n\nTRÍCH ĐOẠN TÀI LIỆU:\n{context_str}\n\nCÂU HỎI: "
                                      "{query_str}\n\nTRẢ LỜI:")
        self.RetrieverQueryEngine = RetrieverQueryEngine

    def answer(self, question):
        from llama_index.core.response_synthesizers import get_response_synthesizer

        t0 = time.time()
        with _lock:
            hits = self.retriever.retrieve(question)
        hits = self._rerank(question, hits, lambda h: h.node.get_content())
        # small-to-big: hand the parent context to the synthesizer
        from llama_index.core.schema import NodeWithScore, TextNode

        ctx, seen, ctx_nodes = [], set(), []
        for h in hits:
            c = h.node.metadata["context"]
            if c in seen:
                continue
            seen.add(c)
            ids = h.node.metadata["node_ids"].split(",")
            ctx.append({"node_ids": ids, "context": c})
            ctx_nodes.append(NodeWithScore(node=TextNode(text=f"[nguồn: {', '.join(ids)}]\n{c}"), score=h.score))
        t_retr = time.time() - t0
        synth = get_response_synthesizer(llm=self.llm, response_mode="compact", text_qa_template=self.qa_tmpl)
        resp = synth.synthesize(question, nodes=ctx_nodes)
        nodes = list(dict.fromkeys(n for h in hits for n in h.node.metadata["node_ids"].split(",")))
        return _timed_result(str(resp), ctx, t_retr, t0, nodes)


def _LIBM25(nodes, k):
    """Minimal LlamaIndex BM25 retriever (llama-index-retrievers-bm25 is a separate package) over bm25s."""
    import bm25s
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore

    class R(BaseRetriever):
        def __init__(self):
            super().__init__()
            self.idx = bm25s.BM25()
            self.idx.index([tok_fold_plus(n.text) for n in nodes], show_progress=False)

        def _retrieve(self, qb):
            d, s = self.idx.retrieve([tok_fold_plus(qb.query_str) or ["_"]], k=min(k, len(nodes)),
                                     show_progress=False)
            return [NodeWithScore(node=nodes[int(i)], score=float(x)) for i, x in zip(d[0], s[0])]

    return R()


# ---------------------------------------------------------------- Haystack

class HaystackRAG(_FrameworkBase):
    framework = "haystack"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        from haystack import Document, Pipeline as HPipeline, component
        from haystack.components.builders import ChatPromptBuilder
        from haystack.components.generators.chat import OpenAIChatGenerator
        from haystack.components.joiners import DocumentJoiner
        from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
        from haystack.dataclasses import ChatMessage
        from haystack.document_stores.in_memory import InMemoryDocumentStore
        from haystack.utils import Secret

        emb, me = self.emb, self

        @component
        class QueryEmbedder:
            @component.output_types(embedding=list[float])
            def run(self, text: str):
                return {"embedding": emb.embed_queries([text])[0].tolist()}

        @component
        class Ranker:
            @component.output_types(documents=list[Document])
            def run(self, query: str, documents: list[Document]):
                return {"documents": me._rerank(query, documents, lambda d: d.content)}

        @component
        class Contexts:
            @component.output_types(context=str, contexts=list)
            def run(self, documents: list[Document]):
                ctx, seen = [], set()
                for d in documents:
                    if d.meta["context"] not in seen:
                        seen.add(d.meta["context"])
                        ctx.append({"node_ids": d.meta["node_ids"], "context": d.meta["context"]})
                return {"context": prompts.format_contexts(ctx), "contexts": ctx}

        store = InMemoryDocumentStore(bm25_algorithm="BM25Okapi")
        vecs = emb.embed_docs([c["text"] for c in self.chunks])
        # InMemoryDocumentStore BM25 tokenizes with a regex; pre-folded text gives it the diacritic-free copies too
        store.write_documents([Document(id=c["chunk_id"], content=" ".join(tok_fold_plus(c["text"])),
                                        embedding=v.tolist(),
                                        meta={"node_ids": c["node_ids"], "context": c["context"], "raw": c["text"]})
                               for c, v in zip(self.chunks, vecs)])
        template = [ChatMessage.from_system(prompts.SYSTEM_QA),
                    ChatMessage.from_user("TRÍCH ĐOẠN TÀI LIỆU:\n{{ context }}\n\nCÂU HỎI: {{ question }}\n\nTRẢ LỜI:")]
        # retrieval pipeline (shared torch models -> run under the lock) and generation pipeline (no lock)
        p = HPipeline()
        p.add_component("qemb", QueryEmbedder())
        p.add_component("dense", InMemoryEmbeddingRetriever(store, top_k=self.depth))
        p.add_component("bm25", InMemoryBM25Retriever(store, top_k=self.depth))
        p.add_component("join", DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=self.depth))
        p.add_component("rank", Ranker())
        p.add_component("ctx", Contexts())
        p.connect("qemb.embedding", "dense.query_embedding")
        p.connect("dense", "join")
        p.connect("bm25", "join")
        p.connect("join.documents", "rank.documents")
        p.connect("rank.documents", "ctx.documents")
        g = HPipeline()
        g.add_component("prompt", ChatPromptBuilder(template=template, required_variables=["context", "question"]))
        g.add_component("llm", OpenAIChatGenerator(api_key=Secret.from_token(API_KEY), model=MODEL,
                                                   api_base_url=BASE_URL, timeout=900,
                                                   generation_kwargs={"temperature": 0, "max_tokens": ANSWER_MAX_TOKENS,
                                                                      "extra_body": EXTRA_BODY}))
        g.connect("prompt.prompt", "llm.messages")
        self.pipe, self.gen = p, g

    def answer(self, question):
        t0 = time.time()
        folded = " ".join(tok_fold_plus(question))
        with _lock:  # retrieval components share torch models that are not thread-safe
            out = self.pipe.run({"qemb": {"text": question}, "bm25": {"query": folded}, "rank": {"query": question}},
                                include_outputs_from={"ctx", "rank"})
        t_retr = time.time() - t0
        gen = self.gen.run({"prompt": {"question": question, "context": out["ctx"]["context"]}})
        ans = gen["llm"]["replies"][0].text
        nodes = list(dict.fromkeys(n for d in out["rank"]["documents"] for n in d.meta["node_ids"]))
        return _timed_result(ans, out["ctx"]["contexts"], t_retr, t0, nodes)


# ---------------------------------------------------------------- DSPy

class DSPyRAG(Pipeline):
    """DSPy module (signature-based prompt) over the shared retriever; optional BootstrapFewShot compile on dev."""

    framework = "dspy"

    def __init__(self, chunking, retriever, top_k=5):
        import dspy

        self.base = NaiveRAG(chunking, retriever, top_k)
        self.name = f"dspy[{chunking}|{retriever}|k{top_k}]"
        self.lm = dspy.LM(f"openai/{MODEL}", api_base=BASE_URL, api_key=API_KEY, temperature=0, max_tokens=ANSWER_MAX_TOKENS,
                          cache=False, extra_body=EXTRA_BODY)

        class AnswerSig(dspy.Signature):
            __doc__ = prompts.SYSTEM_QA
            context: str = dspy.InputField(desc="trích đoạn tài liệu nội bộ")
            question: str = dspy.InputField()
            answer: str = dspy.OutputField(desc="câu trả lời tiếng Việt kèm nguồn [MÃ Điều N]")

        self.program = dspy.Predict(AnswerSig)
        self.dspy = dspy

    def describe(self):
        return {**self.base.describe(), "name": self.name, "framework": self.framework, "technique": "naive (DSPy)"}

    def answer(self, question):
        t0 = time.time()
        rk = self.base.search([question])[0]
        ctx = self.base.select_contexts(rk)
        t_retr = time.time() - t0
        with self.dspy.context(lm=self.lm):
            pred = self.program(context=prompts.format_contexts(ctx), question=question)
        return _timed_result(pred.answer, ctx, t_retr, t0, ranked_nodes(rk, self.base.chunks)[:10])


def strip_think(s):
    return re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()
