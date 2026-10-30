"""Extract the customer-specific fields from a PO PDF with the local Qwen VLM.

Pipeline per file:
  1. render pages + text layer (`pdf.load_pdf`); a garbled text layer (fonts without a Unicode map, e.g. AEON)
     is dropped so it cannot mislead the model; pages matching `skip_pages_containing` are ignored
  2. pick the customer: given explicitly, else from file/folder name or text (`spec.detect_customer`),
     else ask the VLM to choose among the known customers from page 1
  3. ask the VLM for exactly that customer's fields, constrained by a JSON schema whose keys are the field
     names (llama.cpp turns it into a grammar, so other keys cannot be produced). The first page is asked for
     every field; each further page only for the line-item tables, whose rows are concatenated
  4. post-process: keep only the spec's fields, in spec order, null when absent, normalize numbers/dates,
     apply `offset_days`; then look the store code up in the DMS Excel (`dms.lookup`)

CLI:  python -m bibica_rag.po.extract "PO Extraction/" --out results/po
"""
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path

from bibica_rag.po.normalize import clean_text, normalize_value, parse_date, slug
from bibica_rag.po.pdf import PdfPages, image_data_url, load_pdf
from bibica_rag.po.spec import (PO_DIR, CustomerSpec, Field, default_spec_path, detect_customer, find_dms_file,
                                load_specs)

ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = """Bạn là chuyên viên nhập liệu của Bibica, đọc đơn đặt hàng (PO / Purchase Order) do siêu thị gửi.
Nhiệm vụ: trích xuất ĐÚNG và CHỈ các trường được yêu cầu, trả về một đối tượng JSON.
Quy tắc:
- Chỉ dùng thông tin nhìn thấy trong tài liệu; KHÔNG suy đoán, KHÔNG bịa. Không tìm thấy -> null.
- Không thêm trường nào khác ngoài danh sách; giữ nguyên tên trường (key) như danh sách.
- Văn bản / mã: chép nguyên văn từng ký tự như trong tài liệu (mã số, mã vạch, chữ hoa/thường, dấu gạch).
- Số (number): trả về số JSON, bỏ dấu phân cách hàng nghìn và đơn vị (vd "1,308.00" -> 1308).
- Ngày (date): định dạng YYYY-MM-DD. Tài liệu Việt Nam ghi ngày trước tháng (05/03/2025 = 2025-03-05).
- Bảng (table): mọi dòng hàng theo đúng thứ tự trong tài liệu, mỗi dòng chỉ gồm các cột được yêu cầu;
  không lấy dòng tổng cộng."""


@dataclass
class ExtractionResult:
    file: str
    customer: str | None
    customer_reason: str
    fields: dict = field(default_factory=dict)
    dms: dict = field(default_factory=dict)  # Mã DMS / Tên siêu thị DMS / Đơn vị tính DMS from the Excel
    missing: list[str] = field(default_factory=list)
    dropped_keys: list[str] = field(default_factory=list)  # keys the model returned that are not in the spec
    pages_used: list[int] = field(default_factory=list)  # 1-based page numbers sent to the VLM
    n_pages: int = 0
    text_layer: str = ""  # used / garbled / none / disabled
    elapsed_s: float = 0.0
    raw_responses: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- text layer

_PLAIN = re.compile(r"[0-9A-Za-zÀ-ỹ\s.,:;/\-()%#&+'\"*_=]")


def text_is_usable(text: str) -> bool:
    """False for empty layers and for glyph-id garbage such as '[GE67 I>\\E $HUB <BO>' (fonts without ToUnicode)."""
    t = re.sub(r"\s+", " ", text or "").strip()
    if len(t) < 30:
        return False
    plain = sum(bool(_PLAIN.match(c)) for c in t) / len(t)
    tokens = re.findall(r"\S+", t)
    wordlike = sum(bool(re.fullmatch(r"[A-Za-zÀ-ỹ]{2,}[.,:]?|[\d.,/:\-]+", w)) for w in tokens) / len(tokens)
    return plain > 0.97 and wordlike > 0.6


# ---------------------------------------------------------------- prompt + schema

def _field_line(f: Field, indent: str = "- ") -> str:
    line = f'{indent}"{f.name}" ({f.type})'
    if f.description:
        line += f": {f.description}"
    if f.example:
        line += f" (ví dụ: {f.example})"
    if f.type == "table" and f.columns:
        line += "\n" + "\n".join(_field_line(c, "    - cột ") for c in f.columns)
    return line


def build_prompt(spec: CustomerSpec, pdf_text: str = "", fields: list[Field] | None = None,
                 page_note: str = "") -> str:
    fields = spec.fields if fields is None else fields
    parts = [f"Khách hàng: {spec.name}"]
    if spec.notes:
        parts.append(f"Ghi chú riêng cho khách hàng này: {spec.notes}")
    if page_note:
        parts.append(page_note)
    parts.append("Các trường cần trích xuất:\n" + "\n".join(_field_line(f) for f in fields))
    if pdf_text.strip():
        parts.append("Lớp văn bản trích tự động từ PDF (chỉ để tham khảo, có thể thiếu hoặc sai thứ tự; "
                     "ưu tiên những gì nhìn thấy trên ảnh):\n<pdf_text>\n" + pdf_text[:12000] + "\n</pdf_text>")
    parts.append("Trả về DUY NHẤT một đối tượng JSON với đúng các key ở trên.")
    return "\n\n".join(parts)


def _json_type(f: Field) -> dict:
    if f.type == "table":
        item = {"type": "object"}
        if f.columns:
            item.update(properties={c.name: _json_type(c) for c in f.columns},
                        required=[c.name for c in f.columns], additionalProperties=False)
        return {"type": "array", "items": item}
    if f.type == "number":
        return {"type": ["number", "string", "null"]}
    return {"type": ["string", "null"]}


def build_schema(spec: CustomerSpec, fields: list[Field] | None = None) -> dict:
    fields = spec.fields if fields is None else fields
    return {"type": "object", "properties": {f.name: _json_type(f) for f in fields},
            "required": [f.name for f in fields], "additionalProperties": False}


# ---------------------------------------------------------------- response parsing

def parse_json_response(text: str) -> dict:
    """Parse the model output, tolerating <think> blocks, ``` fences and text around the object."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Model did not return JSON: {text[:200]!r}")
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
    return data


def _lookup(data: dict, name: str):
    """Value for `name`: exact key first, then accent/case/punctuation-insensitive key."""
    if name in data:
        return name, data[name]
    s = slug(name)
    for k, v in data.items():
        if slug(k) == s:
            return k, v
    return None, None


def postprocess(spec: CustomerSpec, data: dict) -> tuple[dict, list[str], list[str]]:
    """Keep only the spec's fields (in spec order). Returns (fields, missing, dropped_keys)."""
    out, used = {}, set()
    for f in spec.fields:
        key, v = _lookup(data, f.name)
        if key is not None:
            used.add(key)
        out[f.name] = _normalize_field(f, v)
    missing = [k for k, v in out.items() if v is None or v == []]
    dropped = [k for k in data if k not in used]
    return out, missing, dropped


def _normalize_field(f: Field, v):
    if f.type != "table":
        if isinstance(v, (list, dict)):  # model nested a scalar; flatten instead of losing it
            v = json.dumps(v, ensure_ascii=False) if v else None
        v = normalize_value(v, f.type)
        if f.type == "date" and f.offset_days and v is not None:
            d = parse_date(v)
            v = (d + timedelta(days=f.offset_days)).isoformat() if d else v
        return v
    if v is None:
        return []
    if isinstance(v, dict):
        v = [v]
    rows = []
    for row in v if isinstance(v, list) else []:
        if not isinstance(row, dict):
            continue
        if f.columns:
            r = {c.name: _normalize_field(c, _lookup(row, c.name)[1]) for c in f.columns}
        else:
            r = {clean_text(k): normalize_value(x, "text") for k, x in row.items()}
        if any(x not in (None, "", []) for x in r.values()):
            rows.append(r)
    return rows


# ---------------------------------------------------------------- VLM calls

def _chat(client, model: str, content: list, schema: dict | None, max_tokens: int) -> str:
    from bibica_rag.llm import NO_THINK

    kwargs = dict(model=model, temperature=0, max_tokens=max_tokens, extra_body=NO_THINK,
                  messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}])
    if schema is not None:
        try:
            resp = client.chat.completions.create(
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "po", "strict": True, "schema": schema}}, **kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 — older llama-server builds reject json_schema; retry unconstrained
            if "response_format" not in str(e) and "schema" not in str(e).lower() and "grammar" not in str(e):
                raise
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def _content(prompt: str, images: list) -> list:
    content = [{"type": "image_url", "image_url": {"url": image_data_url(img)}} for img in images]
    return content + [{"type": "text", "text": prompt}]


def classify_customer(client, model: str, specs: dict[str, CustomerSpec], pages: PdfPages) -> str | None:
    names = list(specs)
    prompt = ("Đây là đơn đặt hàng gửi tới Bibica. Khách hàng (siêu thị / bên đặt hàng) là ai? Chọn đúng một tên "
              "trong danh sách, hoặc null nếu không có trong danh sách:\n" +
              "\n".join(f"- {n}" + (f" ({', '.join(specs[n].aliases)})" if specs[n].aliases else "") for n in names)
              + '\nTrả về JSON: {"customer": <tên>}')
    schema = {"type": "object", "properties": {"customer": {"enum": names + [None]}}, "required": ["customer"],
              "additionalProperties": False}
    raw = _chat(client, model, _content(prompt, pages.images[:1]), schema, max_tokens=200)
    try:
        name = parse_json_response(raw).get("customer")
    except ValueError:
        return None
    if name in specs:
        return name
    for n in names:  # tolerate small spelling differences
        if name and slug(n) == slug(name):
            return n
    return None


def select_pages(spec: CustomerSpec, pages: PdfPages) -> list[int]:
    """0-based indexes of the pages to send (all, minus those matching skip_pages_containing)."""
    keep = []
    for i, t in enumerate(pages.texts):
        if any(slug(m) in slug(t) for m in spec.skip_pages_containing):
            continue
        keep.append(i)
    return keep or [0]


def extract_po(pdf_path: str | Path, specs: dict[str, CustomerSpec], customer: str | None = None,
               client=None, model: str | None = None, max_pages: int = 20, dpi: int = 150,
               use_text_layer: bool = True, max_tokens: int = 4096, pages: PdfPages | None = None,
               dms_mapping: dict | None = None) -> ExtractionResult:
    from bibica_rag import llm

    client = client or llm.get_client()
    model = model or llm.MODEL
    t0 = time.time()
    res = ExtractionResult(file=str(pdf_path), customer=None, customer_reason="")
    try:
        pages = pages or load_pdf(pdf_path, dpi=dpi, max_pages=max_pages)
        res.n_pages = pages.n_pages
        all_text = "\n".join(pages.texts)
        usable = text_is_usable(all_text)
        res.text_layer = ("disabled" if not use_text_layer else "used" if usable else
                          "garbled" if all_text.strip() else "none")
        if customer:
            if customer not in specs:
                raise KeyError(f"Khách hàng '{customer}' không có trong file cấu hình")
            res.customer, res.customer_reason = customer, "chọn thủ công"
        else:
            res.customer, res.customer_reason = detect_customer(specs, pdf_path, all_text if usable else "")
            if res.customer is None and specs:
                res.customer = classify_customer(client, model, specs, pages)
                res.customer_reason = "VLM nhận diện từ trang 1" if res.customer else res.customer_reason
        if res.customer is None:
            raise LookupError("Không nhận diện được khách hàng — hãy chọn khách hàng thủ công")
        spec = specs[res.customer]
        idx = select_pages(spec, pages) if usable else list(range(len(pages.images)))
        res.pages_used = [i + 1 for i in idx]
        tables = [f for f in spec.fields if f.type == "table"]
        merged: dict = {}
        for n, i in enumerate(idx):
            if n > 0 and not tables:
                break
            fields = spec.fields if n == 0 else tables
            note = "" if len(idx) == 1 else (
                f"Ảnh là trang {i + 1}/{pages.n_pages} của PO." +
                ("" if n == 0 else " Chỉ lấy các dòng hàng xuất hiện trên trang này (trả về [] nếu không có)."))
            text = pages.texts[i] if use_text_layer and usable else ""
            raw = _chat(client, model, _content(build_prompt(spec, text, fields, note), [pages.images[i]]),
                        build_schema(spec, fields), max_tokens)
            res.raw_responses.append(raw)
            data = parse_json_response(raw)
            for f in fields:
                k, v = _lookup(data, f.name)
                if f.type == "table":
                    merged.setdefault(f.name, [])
                    merged[f.name] += v if isinstance(v, list) else ([v] if isinstance(v, dict) else [])
                elif n == 0:
                    merged[f.name] = v
            res.dropped_keys += [k for k in data if not any(slug(k) == slug(f.name) for f in fields)]
        res.fields, res.missing, _ = postprocess(spec, merged)
        if spec.dms_sheet and dms_mapping is not None:
            from bibica_rag.po.dms import lookup

            res.dms = lookup(dms_mapping, spec.dms_sheet, res.fields.get(spec.code_field))
    except Exception as e:  # noqa: BLE001 — one bad file must not stop a batch
        res.error = f"{e.__class__.__name__}: {e}"
    res.elapsed_s = round(time.time() - t0, 2)
    return res


def iter_pdfs(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() == ".pdf")


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Extract customer-specific fields from PO PDFs with the local VLM")
    ap.add_argument("inputs", nargs="*", default=[str(PO_DIR)], help="PDF files or folders")
    ap.add_argument("--spec", help="YAML/Excel field spec (default: config/po_customers.yaml)")
    ap.add_argument("--dms", help="store-code Excel (default: the DS_SIEU_THI_VA_MA_DMS file in 'PO Extraction/')")
    ap.add_argument("--customer", help="force the customer instead of auto-detection")
    ap.add_argument("--out", default=str(ROOT / "results" / "po"), help="output folder (one JSON per PDF)")
    ap.add_argument("--max-pages", type=int, default=20)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--no-text-layer", action="store_true", help="send only the page images")
    args = ap.parse_args(argv)

    spec_path = args.spec or default_spec_path()
    if not spec_path:
        sys.exit("No spec file: pass --spec")
    specs = load_specs(spec_path)
    dms_path = args.dms or find_dms_file()
    mapping = None
    if dms_path:
        from bibica_rag.po.dms import load_dms_mapping

        mapping = load_dms_mapping(dms_path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for inp in args.inputs:
        for pdf in iter_pdfs(inp):
            r = extract_po(pdf, specs, customer=args.customer, max_pages=args.max_pages, dpi=args.dpi,
                           use_text_layer=not args.no_text_layer, dms_mapping=mapping)
            (out / f"{pdf.stem}.json").write_text(json.dumps(r.to_dict(), ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
            status = r.error or f"{len(r.fields) - len(r.missing)}/{len(r.fields)} fields"
            print(f"{pdf.name}: [{r.customer}] {status} pages={r.pages_used} ({r.elapsed_s}s)")


if __name__ == "__main__":
    main()
