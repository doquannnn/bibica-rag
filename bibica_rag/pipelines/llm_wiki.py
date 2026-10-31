"""LLM-Wiki (Karpathy's pattern): an LLM compiles the documents into an interlinked topic wiki once; at query time the
LLM reads the wiki index, picks the relevant pages and answers from them (no chunk retrieval).

Variants: data/wiki/claude (compiled by Claude) and data/wiki/qwen (compiled by the local Qwen, `python -m
bibica_rag.pipelines.llm_wiki build`).

Page format: line 1 `# Title`, line 2 `Nguồn: <node ids>`, line 3 `Liên quan: <slugs>`, then the body.
"""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from .. import prompts
from ..corpus import DATA, load_nodes
from ..llm import ask, chat
from .base import Pipeline

WIKI = DATA / "wiki"
PAGE_BUDGET = 12000  # chars of wiki pages handed to the generator

SELECT = """Bạn đang tra cứu một wiki nội bộ về chính sách nhân sự của Công ty Bibica. Mục lục wiki:

{index}

Câu hỏi của nhân viên: {q}

Chọn tối đa {k} trang (theo mã trong ngoặc vuông) có khả năng chứa câu trả lời nhất, trang quan trọng nhất trước. Nếu câu hỏi liên quan nhiều chủ đề hoặc so sánh tài liệu, chọn đủ các trang cần thiết.
Chỉ trả lời JSON: {{"pages": ["ma-trang-1", "ma-trang-2"]}}"""


def load_wiki(variant):
    d = WIKI / variant
    pages = {}
    for p in sorted(d.glob("*.md")):
        if p.name == "index.md":
            continue
        lines = p.read_text(encoding="utf-8").split("\n")
        src = lines[1].split(":", 1)[1] if len(lines) > 1 and lines[1].startswith("Nguồn") else ""
        pages[p.stem] = {"title": lines[0].lstrip("# ").strip(), "text": "\n".join(lines),
                         "nodes": [s.strip() for s in src.split(",") if s.strip()]}
    return (d / "index.md").read_text(encoding="utf-8"), pages


class LLMWiki(Pipeline):
    framework = "python"

    def __init__(self, variant="claude", k=3):
        self.variant, self.k = variant, k
        self.index, self.pages = load_wiki(variant)
        self.name = f"llm-wiki[{variant}|k{k}]"

    def describe(self):
        return {"name": self.name, "framework": self.framework, "chunking": f"wiki pages ({self.variant})",
                "retriever": "LLM index navigation", "top_k": self.k, "technique": f"LLM-Wiki ({self.variant})",
                "n_pages": len(self.pages)}

    def select(self, question):
        r = ask(SELECT.format(index=self.index, q=question, k=self.k), max_tokens=120, json_mode=True)
        try:
            slugs = json.loads(r["text"]).get("pages", [])
        except (json.JSONDecodeError, AttributeError):
            slugs = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)+", r["text"])
        return [s for s in dict.fromkeys(slugs) if s in self.pages][: self.k]

    def answer(self, question):
        t0 = time.time()
        slugs = self.select(question)
        ctx, used = [], 0
        for s in slugs:
            txt = self.pages[s]["text"]
            if ctx and used + len(txt) > PAGE_BUDGET:
                txt = txt[: max(0, PAGE_BUDGET - used)]
            if not txt:
                break
            ctx.append({"node_ids": self.pages[s]["nodes"], "context": txt})
            used += len(txt)
        t_sel = time.time() - t0
        r = chat(prompts.qa_messages(question, ctx))
        return {"answer": r["text"], "contexts": ctx,
                "retrieved_nodes": list(dict.fromkeys(n for c in ctx for n in c["node_ids"]))[:10],
                "timings": {"retrieval_s": t_sel, "ttft_s": t_sel + r["ttft_s"], "total_s": time.time() - t0},
                "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"], "llm_calls": 2,
                "pages": slugs}


# ---------------------------------------------------------------- building the wiki with the local Qwen

PLAN = """Dưới đây là mục lục (mã node — tiêu đề) của 3 tài liệu nhân sự của Công ty Bibica (QCL = Quy chế tiền lương tiền thưởng, NQLD = Nội quy lao động, TULD = Thỏa ước lao động tập thể):

{toc}

Hãy thiết kế một wiki theo CHỦ ĐỀ mà nhân viên hay tra cứu (ví dụ: nghỉ phép năm, làm thêm giờ, trả lương, kỷ luật...), gồm 15-25 trang. Mỗi trang gom các node liên quan từ cả 3 tài liệu. Mỗi node phải thuộc ít nhất một trang.
Chỉ trả lời JSON: {{"pages": [{{"slug": "nghi-phep-nam", "title": "Nghỉ phép năm", "nodes": ["NQLD/Điều 7", "QCL/Điều 19"]}}]}}"""

WRITE = """Bạn biên soạn một trang wiki nội bộ về chủ đề "{title}" cho nhân viên Công ty Bibica, CHỈ dựa trên các trích đoạn tài liệu dưới đây.

{sources}

Yêu cầu:
- Tổng hợp ĐẦY ĐỦ mọi quy định liên quan đến chủ đề trong các trích đoạn; giữ NGUYÊN mọi con số, tỷ lệ %, thời hạn, mức tiền, điều kiện, danh sách, hình thức xử lý.
- Ghi nguồn sau mỗi ý theo dạng [QCL Điều 12], [NQLD Điều 7], [QCL Phụ lục 2].
- Nếu các tài liệu cùng quy định một vấn đề, nêu rõ chỗ giống và chỗ khác nhau.
- Không thêm kiến thức bên ngoài. Viết tiếng Việt, dạng Markdown (có thể dùng gạch đầu dòng, bảng).
Chỉ in ra nội dung trang (không in tiêu đề)."""

SUMMARY = """Tóm tắt trang wiki sau trong MỘT câu tiếng Việt (tối đa 40 từ), liệt kê các ý chính:

{page}"""


def build_qwen_wiki(workers=4):
    nodes = list(load_nodes())
    by_id = {n["node_id"]: n for n in nodes}
    toc = "\n".join(f"{n['node_id']} — {n['title'][:120]}" for n in nodes)
    r = ask(PLAN.format(toc=toc), max_tokens=8000, json_mode=True)
    plan = json.loads(r["text"])["pages"]
    plan = [p for p in plan if p.get("slug") and p.get("nodes")]
    for p in plan:
        p["slug"] = re.sub(r"[^a-z0-9-]+", "-", p["slug"].lower()).strip("-")
        p["nodes"] = [n for n in p["nodes"] if n in by_id]
    covered = {n for p in plan for n in p["nodes"]}
    missing = [n["node_id"] for n in nodes if n["node_id"] not in covered]
    if missing:  # never silently drop content: orphan nodes get their own catch-all page
        plan.append({"slug": "cac-quy-dinh-khac", "title": "Các quy định khác", "nodes": missing})
    print(f"plan: {len(plan)} pages, {len(missing)} orphan nodes -> catch-all page", flush=True)

    out = WIKI / "qwen"
    out.mkdir(parents=True, exist_ok=True)

    def write(p):
        src = "\n\n".join(f"[{nid}] {by_id[nid]['title']}\n{by_id[nid]['text']}" for nid in p["nodes"])
        body = ask(WRITE.format(title=p["title"], sources=src), max_tokens=2500)["text"]
        body = re.sub(r"<think>.*?</think>", "", body, flags=re.S).strip()
        related = [q["slug"] for q in plan if q is not p and set(q["nodes"]) & set(p["nodes"])][:5]
        text = f"# {p['title']}\nNguồn: {', '.join(p['nodes'])}\nLiên quan: {', '.join(related)}\n\n{body}\n"
        (out / f"{p['slug']}.md").write_text(text, encoding="utf-8")
        summ = ask(SUMMARY.format(page=text[:6000]), max_tokens=120)["text"].strip().replace("\n", " ")
        return f"- [{p['slug']}] {p['title']} — {summ} (nguồn: {', '.join(p['nodes'])})"

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        lines = list(ex.map(write, plan))
    (out / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(plan)} pages in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build_qwen_wiki()
