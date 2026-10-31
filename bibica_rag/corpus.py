"""Load the parsed corpus (nodes.jsonl) and image captions."""
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PARSED = DATA / "parsed"
CACHE = DATA / "cache"
CAPTIONS = PARSED / "image_captions.json"

DOC_TITLES = {
    "QCL": "Quy chế tiền lương, tiền thưởng Bibica năm 2024",
    "NQLD": "Nội quy lao động 2024",
    "TULD": "Thỏa ước lao động tập thể 2024",
}


@lru_cache(maxsize=1)
def load_nodes() -> tuple:
    """All nodes in document order; image placeholders replaced by VLM captions when available."""
    captions = json.loads(CAPTIONS.read_text(encoding="utf-8")) if CAPTIONS.exists() else {}
    nodes = []
    for line in (PARSED / "nodes.jsonl").read_text(encoding="utf-8").splitlines():
        n = json.loads(line)

        def repl(m):
            name = m.group(1)
            cap = captions.get(name)
            return f"[Hình ảnh {name}: {cap}]" if cap else m.group(0)

        n["text"] = re.sub(r"\[HÌNH ẢNH: ([^\]]+)\]", repl, n["text"])
        nodes.append(n)
    return tuple(nodes)


def node_header(n: dict) -> str:
    parts = [n["doc_title"]]
    if n.get("chapter"):
        parts.append(n["chapter"])
    parts.append(n["title"])
    return " > ".join(parts)


def node_by_id() -> dict:
    return {n["node_id"]: n for n in load_nodes()}


def full_corpus_text() -> str:
    """Whole corpus as one Markdown string (used by long-context / CAG baselines)."""
    out = []
    for doc, title in DOC_TITLES.items():
        out.append(f"# {title} (mã {doc})")
        for n in load_nodes():
            if n["doc"] == doc:
                out.append(f"## [{n['node_id']}] {n['title']}\n{n['text']}")
    return "\n\n".join(out)


def strip_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")
