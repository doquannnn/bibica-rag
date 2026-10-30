"""Parse the Bibica .docx files into Markdown + a list of structural nodes (Điều / Phụ lục).

Word auto-numbering ("1.", "a.", "i.") is not stored in paragraph text, so we emulate it from
numbering.xml. Tables are rendered as Markdown; images are extracted and left as placeholders.
"""
import json
import re
import unicodedata
from pathlib import Path

import docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[2]
DOCX_DIR = ROOT / "data" / "docx"
OUT_DIR = ROOT / "data" / "parsed"

DOCS = {
    "QCL": ("4_QUY_CHE_TIEN_LUONG_TIEN_THUONG_BIBICA_NAM_2024_KHCNS_240426.docx",
            "Quy chế tiền lương, tiền thưởng Bibica năm 2024"),
    "NQLD": ("5. NOI QUI LAO DONG 2024 FINAL.docx", "Nội quy lao động 2024"),
    "TULD": ("6. THOA UOC LD TAP THE 2024 FINAL.docx", "Thỏa ước lao động tập thể 2024"),
}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# ---------------------------------------------------------------- numbering

def _roman(n: int) -> str:
    vals = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
            (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


def _fmt(n: int, kind: str) -> str:
    if kind == "decimal":
        return str(n)
    if kind == "decimalZero":
        return f"{n:02d}"
    if kind == "lowerLetter":
        return chr(ord("a") + (n - 1) % 26)
    if kind == "upperLetter":
        return chr(ord("A") + (n - 1) % 26)
    if kind == "lowerRoman":
        return _roman(n)
    if kind == "upperRoman":
        return _roman(n).upper()
    return str(n)


class Numbering:
    def __init__(self, document):
        self.abstract = {}  # abstractNumId -> {ilvl: (start, fmt, text)}
        self.nums = {}  # numId -> (abstractId, {ilvl: startOverride})
        self.counters = {}  # numId -> {ilvl: value}
        part = document.part.numbering_part if hasattr(document.part, "numbering_part") else None
        if part is None:
            return
        el = part.element
        for an in el.findall(qn("w:abstractNum")):
            levels = {}
            for lvl in an.findall(qn("w:lvl")):
                ilvl = int(lvl.get(qn("w:ilvl")))
                start = lvl.find(qn("w:start"))
                nf = lvl.find(qn("w:numFmt"))
                lt = lvl.find(qn("w:lvlText"))
                levels[ilvl] = (
                    int(start.get(qn("w:val"))) if start is not None else 1,
                    nf.get(qn("w:val")) if nf is not None else "decimal",
                    lt.get(qn("w:val")) if lt is not None else "",
                )
            self.abstract[an.get(qn("w:abstractNumId"))] = levels
        for num in el.findall(qn("w:num")):
            aid = num.find(qn("w:abstractNumId")).get(qn("w:val"))
            overrides = {}
            for ov in num.findall(qn("w:lvlOverride")):
                so = ov.find(qn("w:startOverride"))
                if so is not None:
                    overrides[int(ov.get(qn("w:ilvl")))] = int(so.get(qn("w:val")))
            self.nums[num.get(qn("w:numId"))] = (aid, overrides)

    def label(self, num_id: str, ilvl: int) -> str:
        if num_id not in self.nums or num_id == "0":
            return ""
        aid, overrides = self.nums[num_id]
        levels = self.abstract.get(aid, {})
        if ilvl not in levels:
            return ""
        cnt = self.counters.setdefault(num_id, {})
        start = overrides.get(ilvl, levels[ilvl][0])
        cnt[ilvl] = cnt.get(ilvl, start - 1) + 1
        for deeper in [k for k in cnt if k > ilvl]:
            del cnt[deeper]
        start_, kind, text = levels[ilvl]
        if kind == "bullet":
            return "-"
        if kind == "none" or not text:
            return ""

        def repl(m):
            lv = int(m.group(1)) - 1
            lv_start = overrides.get(lv, levels.get(lv, (1,))[0])
            val = cnt.get(lv, lv_start)
            return _fmt(val, levels.get(lv, (1, "decimal", ""))[1])

        return re.sub(r"%(\d)", repl, text)


def _num_pr(par: Paragraph, document):
    """Return (numId, ilvl) from the paragraph or its style."""
    pPr = par._p.pPr
    if pPr is not None and pPr.numPr is not None:
        n = pPr.numPr
        return (str(n.numId.val) if n.numId is not None else None,
                int(n.ilvl.val) if n.ilvl is not None else 0)
    style = par.style
    while style is not None:
        spPr = style.element.pPr
        if spPr is not None and spPr.numPr is not None:
            n = spPr.numPr
            return (str(n.numId.val) if n.numId is not None else None,
                    int(n.ilvl.val) if n.ilvl is not None else 0)
        style = style.base_style
    return None


# ---------------------------------------------------------------- body walk

def _iter_block_items(document):
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _clean(s: str) -> str:
    s = nfc(s).replace("\xa0", " ").replace("\t", " ")
    return re.sub(r"[ ]{2,}", " ", s).strip()


def _table_md(table: Table) -> str:
    rows = []
    for r in table.rows:
        cells = []
        prev_tc = None
        for c in r.cells:
            # python-docx repeats horizontally merged cells; keep one copy
            if prev_tc is not None and c._tc is prev_tc:
                continue
            prev_tc = c._tc
            cells.append(_clean(c.text.replace("\n", " ")).replace("|", "/"))
        rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    # drop fully empty rows
    rows = [r for r in rows if any(x for x in r)]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _images_in(par: Paragraph):
    blips = par._p.findall(".//" + qn("a:blip"))
    return [b.get(qn("r:embed")) for b in blips if b.get(qn("r:embed"))]


def extract_lines(path: Path, code: str, img_dir: Path):
    document = docx.Document(str(path))
    numbering = Numbering(document)
    lines = []  # list of (kind, text) kind in {"p","table","image"}
    img_count = 0
    for block in _iter_block_items(document):
        if isinstance(block, Table):
            md = _table_md(block)
            if md:
                lines.append(("table", md))
            continue
        for rid in _images_in(block):
            part = document.part.related_parts.get(rid)
            if part is None:
                continue
            img_count += 1
            ext = Path(part.partname).suffix
            name = f"{code}_image{img_count}{ext}"
            (img_dir / name).write_bytes(part.blob)
            lines.append(("image", name))
        text = _clean(block.text)
        np_ = _num_pr(block, document)
        label = numbering.label(*np_) if np_ and np_[0] else ""
        if not text:
            continue
        if label and not text.startswith(label):
            text = f"{label} {text}"
        lines.append(("p", text))
    return lines


HEAD_CHAPTER = re.compile(r"^(CHƯƠNG|Chương)\s+([IVXLC]+)\b\.?\s*(.*)$")
HEAD_ARTICLE = re.compile(r"^(ĐIỀU|Điều)\s+0*(\d+)\s*[:.]?\s*[:.]?\s*(.*)$")
HEAD_APPENDIX = re.compile(r"^(PHỤ LỤC|Phụ lục)\s+0*(\d+)\s*:?\s*(.*)$")


def build_nodes(code: str, title: str, lines):
    nodes = []
    cur = {"node_id": f"{code}/Mở đầu", "kind": "preamble", "number": 0,
           "title": "Phần mở đầu (căn cứ, thông tin ban hành)", "chapter": "", "parts": []}
    chapter = ""
    pending_chapter = False

    def flush():
        if cur["kind"] != "preamble" and not cur["parts"]:
            # whole article fits on its heading line, e.g. "Điều 24. ...: Theo quy định của Bộ Luật Lao động."
            cur["parts"] = [cur["title"]]
        if cur["parts"]:
            nodes.append(dict(cur))

    for kind, text in lines:
        if kind == "p":
            m_ch = HEAD_CHAPTER.match(text)
            m_ar = HEAD_ARTICLE.match(text)
            m_ap = HEAD_APPENDIX.match(text)
            if m_ch:
                chapter = f"Chương {m_ch.group(2)}" + (f": {m_ch.group(3).strip()}" if m_ch.group(3).strip() else "")
                pending_chapter = not m_ch.group(3).strip()
                continue
            if pending_chapter and not (m_ar or m_ap):
                chapter = f"{chapter}: {text}"
                pending_chapter = False
                continue
            pending_chapter = False
            if m_ar:
                flush()
                n = int(m_ar.group(2))
                cur = {"node_id": f"{code}/Điều {n}", "kind": "article", "number": n,
                       "title": f"Điều {n}. {m_ar.group(3).strip()}".strip(), "chapter": chapter, "parts": []}
                continue
            if m_ap:
                flush()
                n = int(m_ap.group(2))
                cur = {"node_id": f"{code}/Phụ lục {n}", "kind": "appendix", "number": n,
                       "title": f"Phụ lục {n:02d}. {m_ap.group(3).strip()}".strip(), "chapter": "Phụ lục",
                       "parts": []}
                continue
            cur["parts"].append(text)
        elif kind == "table":
            cur["parts"].append(text)
        elif kind == "image":
            cur["parts"].append(f"[HÌNH ẢNH: {text}]")
    flush()

    out = []
    for i, n in enumerate(nodes):
        body = "\n".join(n.pop("parts"))
        out.append({**n, "doc": code, "doc_title": title, "order": i, "text": body})
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = ROOT / "data" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    all_nodes = []
    for code, (fname, title) in DOCS.items():
        lines = extract_lines(DOCX_DIR / fname, code, img_dir)
        nodes = build_nodes(code, title, lines)
        all_nodes += nodes
        md = [f"# {title}"]
        chapter = None
        for n in nodes:
            if n["chapter"] and n["chapter"] != chapter:
                chapter = n["chapter"]
                md.append(f"\n## {chapter}")
            md.append(f"\n### {n['title']}  <!-- {n['node_id']} -->\n\n{n['text']}")
        (OUT_DIR / f"{code}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        arts = [n for n in nodes if n["kind"] == "article"]
        print(code, "nodes:", len(nodes), "articles:", len(arts),
              "max article:", max(a["number"] for a in arts),
              "missing:", sorted(set(range(1, max(a["number"] for a in arts) + 1)) - {a["number"] for a in arts}))
    ids = [n["node_id"] for n in all_nodes]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        print("DUPLICATE node ids:", dup)
    with open(OUT_DIR / "nodes.jsonl", "w", encoding="utf-8") as f:
        for n in all_nodes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
