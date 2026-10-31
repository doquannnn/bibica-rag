"""Chunking strategies. Every strategy returns a list of chunk dicts:

    chunk_id      unique id
    text          text that is indexed (embedded / BM25)
    context       text handed to the LLM when the chunk is retrieved (parent article for small-to-big)
    node_ids      structural nodes (Điều / Phụ lục) the chunk overlaps — used for retrieval metrics
    doc           document code
"""
import re

import numpy as np

from .corpus import DOC_TITLES, load_nodes, node_header

KHOAN = re.compile(r"^\d+\.\s")


def _chunk(cid, text, node_ids, doc, context=None):
    return {"chunk_id": cid, "text": text, "context": context or text, "node_ids": list(dict.fromkeys(node_ids)),
            "doc": doc}


def _doc_stream(doc):
    """Concatenate a document's nodes, remembering each node's character span."""
    pieces, spans, pos = [], [], 0
    for n in load_nodes():
        if n["doc"] != doc:
            continue
        block = f"{n['title']}\n{n['text']}\n\n"
        pieces.append(block)
        spans.append((pos, pos + len(block), n["node_id"]))
        pos += len(block)
    return "".join(pieces), spans


def _nodes_in(spans, a, b):
    return [nid for s, e, nid in spans if s < b and e > a]


# ---------------------------------------------------------------- baselines

def fixed(size=800, overlap=150):
    out = []
    for doc in DOC_TITLES:
        text, spans = _doc_stream(doc)
        start, i = 0, 0
        while start < len(text):
            end = min(len(text), start + size)
            if end < len(text):  # snap to whitespace
                ws = text.rfind(" ", start + size // 2, end)
                end = ws if ws > 0 else end
            out.append(_chunk(f"{doc}-fx{i}", text[start:end].strip(), _nodes_in(spans, start, end), doc))
            i += 1
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return out


def recursive(size=1000, overlap=150):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
                                              separators=["\n\n", "\n", ". ", " ", ""], add_start_index=True)
    out = []
    for doc in DOC_TITLES:
        text, spans = _doc_stream(doc)
        for i, d in enumerate(splitter.create_documents([text])):
            a = d.metadata["start_index"]
            out.append(_chunk(f"{doc}-rc{i}", d.page_content, _nodes_in(spans, a, a + len(d.page_content)), doc))
    return out


def sentence(max_chars=600):
    """Group consecutive lines/sentences up to max_chars without crossing node boundaries."""
    out = []
    for n in load_nodes():
        units = [u for u in re.split(r"(?<=[.;:])\s+|\n", n["text"]) if u.strip()]
        buf, i = [], 0
        for u in units + [None]:
            if u is None or (buf and sum(len(x) for x in buf) + len(u) > max_chars):
                if buf:
                    out.append(_chunk(f"{n['node_id']}#s{i}", f"{n['title']}\n" + " ".join(buf),
                                      [n["node_id"]], n["doc"]))
                    i += 1
                buf = []
            if u is not None:
                buf.append(u)
    return out


def semantic(embed_fn, percentile=85, max_chars=1500):
    """Split each document at lines where consecutive-line embedding distance spikes."""
    out = []
    for doc in DOC_TITLES:
        units = []  # (line, node_id)
        for n in load_nodes():
            if n["doc"] != doc:
                continue
            lines = [n["title"]] + [x for x in n["text"].split("\n") if x.strip()]
            units += [(x, n["node_id"]) for x in lines]
        vecs = embed_fn([u[0] for u in units])
        sims = (vecs[:-1] * vecs[1:]).sum(1)
        thr = np.percentile(1 - sims, percentile)
        buf, i = [], 0
        for k, (line, nid) in enumerate(units):
            buf.append((line, nid))
            size = sum(len(x[0]) for x in buf)
            if k == len(units) - 1 or (1 - sims[k]) > thr or size > max_chars:
                out.append(_chunk(f"{doc}-sem{i}", "\n".join(x[0] for x in buf), [x[1] for x in buf], doc))
                buf, i = [], i + 1
    return out


# ---------------------------------------------------------------- structure-aware

def _clauses(n, max_chars=1400):
    """Split an article into its khoản (top-level "1." items); long tables/blocks are split further."""
    lines = n["text"].split("\n")
    groups, cur = [], []
    for ln in lines:
        if KHOAN.match(ln) and cur:
            groups.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        groups.append(cur)
    final = []
    for g in groups:
        block = "\n".join(g)
        if len(block) <= max_chars:
            final.append(block)
            continue
        part = []
        for ln in g:
            if part and sum(len(x) for x in part) + len(ln) > max_chars:
                final.append("\n".join(part))
                # repeat a table header so each table piece stays readable
                part = [x for x in part[:2] if x.startswith("|")] if ln.startswith("|") else []
            part.append(ln)
        if part:
            final.append("\n".join(part))
    return final


def structure_article(max_chars=2500):
    """One chunk per Điều / Phụ lục; oversized ones split by khoản."""
    out = []
    for n in load_nodes():
        head = node_header(n)
        if len(n["text"]) <= max_chars:
            out.append(_chunk(n["node_id"], f"{head}\n{n['text']}", [n["node_id"]], n["doc"]))
        else:
            for i, c in enumerate(_clauses(n, max_chars=max_chars // 2)):
                out.append(_chunk(f"{n['node_id']}#{i}", f"{head}\n{c}", [n["node_id"]], n["doc"]))
    return out


def structure_clause(header=True, parent=False):
    """One chunk per khoản. header: prefix doc>chapter>article path. parent: LLM receives whole article."""
    out = []
    for n in load_nodes():
        head = node_header(n)
        full = f"{head}\n{n['text']}"
        for i, c in enumerate(_clauses(n, max_chars=900)):
            text = f"{head}\n{c}" if header else c
            out.append(_chunk(f"{n['node_id']}#{i}", text, [n["node_id"]], n["doc"],
                              context=full if parent else text))
    return out


def contextual_llm(base_chunks, contexts: dict):
    """Anthropic-style contextual retrieval: prepend an LLM-written situating sentence to each chunk."""
    out = []
    for c in base_chunks:
        ctx = contexts.get(c["chunk_id"], "")
        out.append({**c, "text": f"{ctx}\n{c['text']}" if ctx else c["text"]})
    return out


STRATEGIES = {
    "fixed500": lambda **k: fixed(500, 100),
    "fixed1000": lambda **k: fixed(1000, 200),
    "recursive1000": lambda **k: recursive(1000, 150),
    "sentence600": lambda **k: sentence(600),
    "semantic": lambda embed_fn=None, **k: semantic(embed_fn),
    "article": lambda **k: structure_article(),
    "clause_noheader": lambda **k: structure_clause(header=False),
    "clause_header": lambda **k: structure_clause(header=True),
    "clause_parent": lambda **k: structure_clause(header=True, parent=True),
}
