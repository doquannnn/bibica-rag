"""Self-checks on one extraction, without ground truth — shown to the person reviewing the order.

A VLM can misread a digit or skip a line without any error, so every result is checked against what can be
verified mechanically:
- required fields present, at least one line item, quantities positive numbers
- EAN-13 check digit for 13-digit product codes (a misread digit almost always breaks it)
- duplicate product codes
- grounding in the PDF text layer when there is a usable one (digital PDFs): each code / PO number / store code
  must appear in the text, and each line's quantity must be printed near its product code
- store code not found in the DMS Excel
Scanned PDFs (Aeon, Mega) have no usable text layer, so only the first checks apply to them.
"""
import re

from bibica_rag.po.normalize import parse_number, slug
from bibica_rag.po.spec import CustomerSpec


def ean13_ok(code: str) -> bool:
    if not re.fullmatch(r"\d{13}", code):
        return True  # not an EAN-13: nothing to check
    digits = [int(c) for c in code]
    total = sum(d * (3 if i % 2 else 1) for i, d in enumerate(digits[:12]))
    return (10 - total % 10) % 10 == digits[12]


def _numbers_in(line: str) -> set[float]:
    out = set()
    for tok in re.findall(r"\d[\d.,]*", line):
        for dc in (False, True):
            x = parse_number(tok, decimal_comma=dc)
            if x is not None:
                out.add(round(x, 4))
    return out


def _grounded(value, text_slug: str) -> bool:
    s = slug(value)
    return not s or s in text_slug or s.lstrip("0") in text_slug


def validate(spec: CustomerSpec, fields: dict, text: str = "", dms: dict | None = None) -> list[str]:
    warns = []
    text_slug = slug(text) if text else ""
    lines = text.splitlines() if text else []
    for f in spec.fields:
        v = fields.get(f.name)
        if f.type != "table":
            if v in (None, ""):
                warns.append(f"Thiếu '{f.name}'")
            elif text and f.type == "text" and not _grounded(v, text_slug):
                warns.append(f"'{f.name}' = {v!r} không tìm thấy trong lớp chữ của PDF")
            continue
        rows = v or []
        if not rows:
            warns.append(f"Không đọc được dòng hàng nào ở '{f.name}'")
            continue
        code_col = next((c for c in f.columns if c.type == "text"), None)
        qty_col = next((c for c in f.columns if c.type == "number"), None)
        seen = {}
        for i, row in enumerate(rows, 1):
            code = str(row.get(code_col.name) or "") if code_col else ""
            qty = row.get(qty_col.name) if qty_col else None
            where = f"Dòng {i}" + (f" ({code})" if code else "")
            if code_col and not code:
                warns.append(f"{where}: thiếu '{code_col.name}'")
            if qty_col and (not isinstance(qty, (int, float)) or qty <= 0):
                warns.append(f"{where}: '{qty_col.name}' không hợp lệ ({qty!r})")
            if code and not ean13_ok(code[-13:] if len(code) > 13 and code.isdigit() else code):
                warns.append(f"{where}: mã vạch sai số kiểm tra EAN-13 — có thể đọc sai một chữ số")
            if code:
                if code in seen:
                    warns.append(f"{where}: trùng mã với dòng {seen[code]}")
                seen.setdefault(code, i)
            if text and code:
                hits = [j for j, ln in enumerate(lines) if slug(code) in slug(ln)]
                if not hits:
                    warns.append(f"{where}: mã không có trong lớp chữ của PDF")
                elif isinstance(qty, (int, float)):
                    near = {x for j in hits for ln in lines[max(0, j - 3):j + 4] for x in _numbers_in(ln)}
                    if round(float(qty), 4) not in near:
                        warns.append(f"{where}: số lượng {qty} không thấy cạnh mã trong lớp chữ của PDF")
    if dms and dms.get("ghi_chu"):
        warns.append(f"DMS: {dms['ghi_chu']}")
    return warns
