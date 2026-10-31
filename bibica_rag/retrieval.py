"""Retrievers (dense, BM25, hybrid, bge-m3 multi-vector) and rerankers."""
import hashlib
import json
import re
from functools import lru_cache

import numpy as np

from .corpus import CACHE, strip_diacritics

# ---------------------------------------------------------------- embedders

EMBEDDERS = {
    # name: (hf id, query prefix/prompt, passage prefix, max_seq_len)
    "bge-m3": ("BAAI/bge-m3", "", "", 1024),
    "vn-embedding": ("AITeamVN/Vietnamese_Embedding", "", "", 1024),
    "qwen3-emb-0.6b": ("Qwen/Qwen3-Embedding-0.6B",
                       "Instruct: Given a question from an employee, retrieve the internal regulation passage "
                       "that answers it\nQuery:", "", 1024),
    "me5-large-instruct": ("intfloat/multilingual-e5-large-instruct",
                           "Instruct: Given a question, retrieve passages from company regulations that answer it\n"
                           "Query: ", "", 512),
    "halong": ("hiieu/halong_embedding", "", "", 512),
}


@lru_cache(maxsize=None)
def get_st_model(hf_id, max_len):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(max(1, torch.get_num_threads()))
    m = SentenceTransformer(hf_id, device="cpu", trust_remote_code=True)
    m.max_seq_length = max_len
    return m


class DenseEmbedder:
    def __init__(self, name):
        self.name = name
        self.hf_id, self.qp, self.pp, self.max_len = EMBEDDERS[name]

    @property
    def model(self):
        return get_st_model(self.hf_id, self.max_len)

    def _cached(self, texts, prefix, kind):
        key = hashlib.sha1(json.dumps([self.name, kind, prefix, texts], ensure_ascii=False).encode()).hexdigest()
        path = CACHE / "emb" / f"{key}.npy"
        if path.exists():
            return np.load(path)
        vecs = self.model.encode([prefix + t for t in texts], batch_size=8, normalize_embeddings=True,
                                 show_progress_bar=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
        return vecs

    def embed_docs(self, texts):
        return self._cached(texts, self.pp, "doc")

    def embed_queries(self, texts):
        return self._cached(texts, self.qp, "query")


# ---------------------------------------------------------------- tokenizers for BM25

def tok_syllable(s):
    return re.findall(r"\w+", s.lower())


def tok_fold(s):
    return re.findall(r"\w+", strip_diacritics(s.lower()))


@lru_cache(maxsize=4096)
def _pyvi(s):
    from pyvi import ViTokenizer

    return ViTokenizer.tokenize(s)


def tok_pyvi(s):
    return [t for t in re.findall(r"[\w_]+", _pyvi(s.lower()))]


def tok_fold_plus(s):
    """Accented syllables + diacritic-free copies, so unaccented queries still match."""
    a = tok_syllable(s)
    return a + [strip_diacritics(x) for x in a]


TOKENIZERS = {"syl": tok_syllable, "fold": tok_fold, "pyvi": tok_pyvi, "foldplus": tok_fold_plus}


# ---------------------------------------------------------------- retrievers

class Retriever:
    """Index a chunk list; search(queries) -> list of ranked (chunk_index, score) lists."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]

    def search(self, queries, k=20):
        raise NotImplementedError


class Dense(Retriever):
    def __init__(self, chunks, embedder: str):
        super().__init__(chunks)
        self.emb = DenseEmbedder(embedder)
        self.mat = self.emb.embed_docs(self.texts)

    def search(self, queries, k=20):
        q = self.emb.embed_queries(list(queries))
        sims = q @ self.mat.T
        out = []
        for row in sims:
            idx = np.argsort(-row)[:k]
            out.append([(int(i), float(row[i])) for i in idx])
        return out


class BM25(Retriever):
    def __init__(self, chunks, tokenizer="syl"):
        import bm25s

        super().__init__(chunks)
        self.tok = TOKENIZERS[tokenizer]
        self.index = bm25s.BM25()
        self.index.index([self.tok(t) for t in self.texts], show_progress=False)

    def search(self, queries, k=20):
        out = []
        for q in queries:
            toks = self.tok(q) or ["_"]
            k_ = min(k, len(self.texts))
            docs, scores = self.index.retrieve([toks], k=k_, show_progress=False)
            out.append([(int(d), float(s)) for d, s in zip(docs[0], scores[0])])
        return out


def rrf(rankings, k=60, top=20):
    scores = {}
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])[:top]


class Hybrid(Retriever):
    def __init__(self, chunks, parts):
        super().__init__(chunks)
        self.parts = parts

    def search(self, queries, k=20):
        res = [p.search(queries, k=max(k, 30)) for p in self.parts]
        return [rrf([r[i] for r in res], top=k) for i in range(len(queries))]


@lru_cache(maxsize=1)
def _bgem3():
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, devices="cpu")


class BGEM3All(Retriever):
    """bge-m3 dense + lexical (sparse) + colbert multi-vector, weighted sum (weights from the paper)."""

    def __init__(self, chunks, weights=(0.4, 0.2, 0.4)):
        super().__init__(chunks)
        self.w = weights
        self.m = _bgem3()
        self.doc = self.m.encode(self.texts, batch_size=8, max_length=1024, return_dense=True,
                                 return_sparse=True, return_colbert_vecs=True)

    def search(self, queries, k=20):
        q = self.m.encode(list(queries), batch_size=8, max_length=256, return_dense=True, return_sparse=True,
                          return_colbert_vecs=True)
        out = []
        dense = q["dense_vecs"] @ self.doc["dense_vecs"].T
        for i in range(len(queries)):
            sc = []
            for j in range(len(self.texts)):
                sp = self.m.compute_lexical_matching_score(q["lexical_weights"][i], self.doc["lexical_weights"][j])
                cb = float(self.m.colbert_score(q["colbert_vecs"][i], self.doc["colbert_vecs"][j]))
                sc.append(self.w[0] * dense[i, j] + self.w[1] * sp + self.w[2] * cb)
            sc = np.array(sc)
            idx = np.argsort(-sc)[:k]
            out.append([(int(t), float(sc[t])) for t in idx])
        return out


# ---------------------------------------------------------------- rerankers

RERANKERS = {
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "vn-reranker": "AITeamVN/Vietnamese_Reranker",
    "qwen3-reranker-0.6b": "Qwen/Qwen3-Reranker-0.6B",
}


@lru_cache(maxsize=None)
def _cross_encoder(hf_id):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(hf_id, device="cpu", max_length=512)


class Qwen3Reranker:
    """Qwen3-Reranker scores relevance as P(yes) of a causal LM (per the model card)."""

    def __init__(self, hf_id="Qwen/Qwen3-Reranker-0.6B"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(hf_id, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(hf_id).eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
                       "the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
                       "<|im_start|>user\n")
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.instr = "Given an employee question, judge whether the regulation passage answers it"

    def predict(self, pairs, batch_size=8):
        scores = []
        for b in range(0, len(pairs), batch_size):
            texts = [f"{self.prefix}<Instruct>: {self.instr}\n<Query>: {q}\n<Document>: {d}{self.suffix}"
                     for q, d in pairs[b:b + batch_size]]
            enc = self.tok(texts, padding=True, truncation=True, max_length=768, return_tensors="pt")
            with self.torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :]
            two = self.torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1)
            scores += self.torch.softmax(two, dim=1)[:, 1].tolist()
        return np.array(scores)


@lru_cache(maxsize=None)
def get_reranker(name):
    if name.startswith("qwen3-reranker"):
        return Qwen3Reranker(RERANKERS[name])
    return _cross_encoder(RERANKERS[name])


class Reranked(Retriever):
    def __init__(self, chunks, base: Retriever, reranker: str, depth=10):
        super().__init__(chunks)
        self.base, self.depth, self.name = base, depth, reranker
        self.rr = get_reranker(reranker)

    def _scores(self, q, idxs):
        key = hashlib.sha1(json.dumps([self.name, q, [self.texts[i] for i in idxs]], ensure_ascii=False)
                           .encode()).hexdigest()
        path = CACHE / "rerank" / f"{key}.npy"
        if path.exists():
            return np.load(path)
        s = np.asarray(self.rr.predict([(q, self.texts[i]) for i in idxs]))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, s)
        return s

    def search(self, queries, k=20):
        base = self.base.search(queries, k=self.depth)
        out = []
        for q, cand in zip(queries, base):
            idxs = [i for i, _ in cand]
            s = self._scores(q, idxs)
            order = np.argsort(-s)[:k]
            out.append([(idxs[o], float(s[o])) for o in order])
        return out


# ---------------------------------------------------------------- factory

def build_retriever(spec: str, chunks):
    """spec grammar: 'dense:<emb>' | 'bm25:<tok>' | 'bgem3all' | 'hybrid:<spec>+<spec>' ; '|rerank:<name>' suffix."""
    rerank = None
    if "|rerank:" in spec:
        spec, rerank = spec.split("|rerank:")
    if spec.startswith("hybrid:"):
        r = Hybrid(chunks, [build_retriever(s, chunks) for s in spec[len("hybrid:"):].split("+")])
    elif spec.startswith("dense:"):
        r = Dense(chunks, spec[len("dense:"):])
    elif spec.startswith("bm25:"):
        r = BM25(chunks, spec[len("bm25:"):])
    elif spec == "bgem3all":
        r = BGEM3All(chunks)
    else:
        raise ValueError(spec)
    if rerank:
        r = Reranked(chunks, r, rerank)
    return r
