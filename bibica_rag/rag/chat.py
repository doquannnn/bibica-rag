"""Baseline RAG chat over the parsed Bibica documents (`data/parsed/nodes.jsonl`).

Retrieval: BM25 over accent-free syllable unigrams + bigrams (so "nghi phep nam" matches "nghỉ phép năm"),
one node (Điều / Phụ lục) per document. Generation: the local Qwen model, answering only from the retrieved
articles and citing their node ids. The benchmark can swap `Retriever` for a stronger one later.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np

from bibica_rag.po.normalize import words

ROOT = Path(__file__).resolve().parents[2]
NODES = ROOT / "data" / "parsed" / "nodes.jsonl"

SYSTEM_PROMPT = """Bạn là trợ lý nhân sự nội bộ của Bibica. Chỉ trả lời dựa trên các trích đoạn tài liệu được cung cấp
(Quy chế tiền lương tiền thưởng — QCL, Nội quy lao động — NQLD, Thỏa ước lao động tập thể — TULD).
- Trả lời bằng tiếng Việt, ngắn gọn, đầy đủ, giữ nguyên con số / tỷ lệ như tài liệu.
- Ghi nguồn sau mỗi ý, ví dụ (QCL Điều 12).
- Nếu các trích đoạn không chứa thông tin cần thiết, trả lời: "Tài liệu nội bộ không có thông tin này." — không suy đoán."""


def tokenize(text: str) -> list[str]:
    w = words(text)
    return w + [f"{a}_{b}" for a, b in zip(w, w[1:])]


@dataclass
class Hit:
    node: dict
    score: float


class Retriever:
    def __init__(self, nodes: list[dict]):
        self.nodes = nodes
        self.bm25 = bm25s.BM25()
        corpus = [tokenize(f"{n.get('title', '')}\n{n.get('chapter', '')}\n{n['text']}") for n in nodes]
        self.vocab = {t: i for i, t in enumerate(sorted({t for doc in corpus for t in doc}))}
        self.bm25.index(bm25s.tokenization.Tokenized(
            ids=[[self.vocab[t] for t in doc] for doc in corpus], vocab=self.vocab), show_progress=False)

    @classmethod
    def from_jsonl(cls, path: str | Path = NODES) -> "Retriever":
        with open(path, encoding="utf-8") as f:
            return cls([json.loads(line) for line in f if line.strip()])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        top = np.argsort(-scores)[:k]
        return [Hit(self.nodes[i], float(scores[i])) for i in top if scores[i] > 0]


def build_messages(question: str, hits: list[Hit], history: list[dict] | None = None) -> list[dict]:
    ctx = "\n\n".join(f"[{h.node['node_id']}] {h.node.get('doc_title', '')} — {h.node.get('title', '')}\n"
                      f"{h.node['text']}" for h in hits) or "(không tìm thấy trích đoạn nào)"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in (history or [])[-6:]:  # short memory for follow-up questions
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": f"Trích đoạn tài liệu:\n{ctx}\n\nCâu hỏi: {question}"})
    return msgs


def answer_stream(question: str, retriever: Retriever, history: list[dict] | None = None, k: int = 5,
                  client=None, model: str | None = None):
    """Yield (partial answer, hits)."""
    from bibica_rag import llm

    client = client or llm.get_client()
    hits = retriever.search(question, k=k)
    stream = client.chat.completions.create(model=model or llm.MODEL, temperature=0.1, max_tokens=1500,
                                            messages=build_messages(question, hits, history), stream=True,
                                            extra_body=llm.NO_THINK)
    text = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
            yield text, hits
    if not text:
        yield "", hits
