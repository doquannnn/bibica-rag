"""Per-customer extraction specs: which fields to extract from a given customer's PO.

Default: `config/po_customers.yaml`, transcribed from the Bibica Excel file (DS_SIEU_THI_VA_MA_DMS_*.xlsx), where
the fields are marked as boxes on PO screenshots and therefore cannot be read from the cells.

The loader also reads field lists written as cells in an Excel file; it detects the most common layouts sheet by
sheet (header names are matched accent/case-insensitively):

1. long    — one row per (customer, field):  | Khách hàng | Thông tin cần trích xuất | Mô tả | Kiểu | Nhóm | Ví dụ |
             (customer cells may be merged / left blank under the first row of each customer)
2. matrix  — one row per field, one column per customer, a mark (x, ✓, 1, có, yes) where the customer needs it:
             | Thông tin | Customer A | Customer B | ...
3. wide    — one row per customer, one column per field, a mark or an example value where needed:
             | Khách hàng | Số PO | Ngày PO | ...
4. sheet   — one sheet per customer (sheet name = customer), a field column (+ optional description/type...)

Only "field" / "customer" columns are required; description, type (text/number/date/table), group (fields of
the same table-like group, e.g. "Chi tiết hàng hóa", become columns of one line-item table), example and
alias (keywords to recognise the customer in a PDF) are optional.

A YAML/JSON file with the same structure as `CustomerSpec.to_dict()` can be used instead of — or next to
(`<excel stem>.yaml`, merged on top) — the Excel file when the auto-detection is not good enough.
Run `python -m bibica_rag.po.spec "PO Extraction/<file>.xlsx"` to print what was understood.
"""
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bibica_rag.po.normalize import clean_text, slug, words

FIELD_TYPES = ("text", "number", "date", "table")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / "config" / "po_customers.yaml"
PO_DIR = ROOT / "PO Extraction"


@dataclass
class Field:
    name: str
    type: str = "text"
    description: str = ""
    example: str = ""
    columns: list["Field"] = field(default_factory=list)  # only for type == "table"
    offset_days: int = 0  # date fields: added after extraction (e.g. delivery date = order date + 5 days)
    date_order: str = "dmy"  # date fields: "mdy" when the PO prints month first (Satra: 11/28/2023)
    decimal_comma: bool = False  # number fields: the PO writes decimals with a comma (Satra "1,000" = 1)

    def to_dict(self) -> dict:
        d = {"name": self.name, "type": self.type}
        if self.description:
            d["description"] = self.description
        if self.example:
            d["example"] = self.example
        if self.offset_days:
            d["offset_days"] = self.offset_days
        if self.date_order != "dmy":
            d["date_order"] = self.date_order
        if self.decimal_comma:
            d["decimal_comma"] = True
        if self.columns:
            d["columns"] = [c.to_dict() for c in self.columns]
        return d

    @classmethod
    def from_dict(cls, d) -> "Field":
        if isinstance(d, str):
            return cls(name=d, type=infer_type(d))
        cols = [cls.from_dict(c) for c in d.get("columns", [])]
        return cls(name=clean_text(d["name"]),
                   type=parse_type(d.get("type"), d["name"]) if not cols else "table",
                   description=clean_text(d.get("description", "") or ""),
                   example=clean_text(d.get("example", "") or ""), columns=cols,
                   offset_days=int(d.get("offset_days", 0) or 0),
                   date_order=str(d.get("date_order", "dmy") or "dmy").lower(),
                   decimal_comma=bool(d.get("decimal_comma", False)))


@dataclass
class CustomerSpec:
    name: str
    fields: list[Field] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    notes: str = ""
    skip_pages_containing: list[str] = field(default_factory=list)  # pages whose text has one of these are ignored
    dms_sheet: str = ""  # sheet of the DMS code Excel holding this customer's stores
    code_field: str = ""  # field holding the store code to look up in that sheet

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    def to_dict(self) -> dict:
        d = {"name": self.name, "fields": [f.to_dict() for f in self.fields]}
        if self.aliases:
            d["aliases"] = self.aliases
        if self.notes:
            d["notes"] = self.notes
        for k in ("skip_pages_containing", "dms_sheet", "code_field"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        return d

    @classmethod
    def from_dict(cls, d) -> "CustomerSpec":
        return cls(name=clean_text(d["name"]), fields=[Field.from_dict(f) for f in d.get("fields", [])],
                   aliases=[clean_text(a) for a in d.get("aliases", [])], notes=clean_text(d.get("notes", "") or ""),
                   skip_pages_containing=list(d.get("skip_pages_containing", []) or []),
                   dms_sheet=d.get("dms_sheet", "") or "", code_field=d.get("code_field", "") or "")


# ---------------------------------------------------------------- types

_NUMBER_HINTS = ("soluong", "dongia", "thanhtien", "tongtien", "tongcong", "tonggiatri", "giatri", "vat", "thue",
                 "chietkhau", "quantity", "qty", "price", "amount", "total", "tax", "discount")
_DATE_HINTS = ("ngay", "date", "thoigiangiao", "hangiao", "deadline", "eta", "etd")


def infer_type(name: str) -> str:
    s = slug(name)
    # "Số PO", "Mã hàng", "Địa chỉ giao hàng" ... are identifiers / text even if they contain a hint word
    if s.startswith(("so", "ma", "diachi", "noigiao", "dvt", "donvi")) and not s.startswith("soluong"):
        return "text"
    if s.startswith("sl") or s.startswith("qty"):
        return "number"
    if any(h in s for h in _DATE_HINTS):
        return "date"
    if any(h in s for h in _NUMBER_HINTS):
        return "number"
    return "text"


def parse_type(raw, name: str) -> str:
    s = slug(raw or "")
    if not s:
        return infer_type(name)
    if s in FIELD_TYPES:
        return s
    if any(k in s for k in ("bang", "table", "list", "danhsach", "array")):
        return "table"
    if any(k in s for k in ("ngay", "date", "thoigian", "time")):
        return "date"
    if any(k in s for k in ("so", "number", "numeric", "int", "float", "decimal", "tien", "currency")):
        return "number"
    return "text"


# ---------------------------------------------------------------- header detection

# role -> slug prefixes (checked in this order, so "Mô tả thông tin" is a description, not a field)
_ROLES = {
    "customer": ("khachhang", "tenkhachhang", "customer", "buyer", "client", "doitac", "tenkh", "makh"),
    "description": ("mota", "diengiai", "ghichu", "description", "note", "huongdan", "vitri", "giaithich"),
    "type": ("kieudulieu", "kieu", "loaidulieu", "dinhdang", "type", "datatype", "format"),
    "group": ("nhom", "bang", "group", "table", "section"),
    "example": ("vidu", "example", "mau", "sample", "giatrimau"),
    "alias": ("tukhoa", "alias", "keyword", "nhandien", "tengoikhac"),
    "field": ("thongtin", "truong", "tentruong", "field", "noidung", "item", "chitieu", "attribute", "key",
              "tenthongtin", "cot", "column"),
}
_SHORT_CUSTOMER = ("kh",)
_IGNORE = ("stt", "no", "tt", "#")

_MARK_YES = {"x", "v", "1", "co", "yes", "y", "true", "ok", "can", "required", "bat buoc", "batbuoc"}
_MARK_NO = {"", "0", "khong", "no", "n", "false", "-", "none", "nan"}
_CHECKS = ("✓", "✔", "☑", "√")
_TABLE_GROUP_HINTS = ("chitiet", "hanghoa", "sanpham", "mathang", "danhsach", "lineitem", "item", "bang",
                      "table", "dong", "detail")


def _role(cell) -> str | None:
    s = slug(cell)
    if not s or s in _IGNORE:
        return None
    if s in _SHORT_CUSTOMER:
        return "customer"
    for role, prefixes in _ROLES.items():
        if s.startswith(prefixes):
            return role
    for role, prefixes in _ROLES.items():
        if any(p in s for p in prefixes if len(p) >= 5):
            return role
    return None


def _mark(v) -> bool | None:
    """True/False for check-mark cells, None if the cell holds real content."""
    s = clean_text(v).lower() if v is not None else ""
    if s in _MARK_NO:
        return False
    if s in _MARK_YES or any(c in s for c in _CHECKS) or slug(s) in _MARK_YES:
        return True
    return None


def _cells(row) -> list[str]:
    return ["" if v is None or (isinstance(v, float) and v != v) else clean_text(v) for v in row]


def _find_header(rows: list[list[str]], max_scan: int = 20):
    """First row naming a field/customer column; a lone cell (e.g. a merged title) only as a fallback."""
    fallback = (None, {})
    for i, row in enumerate(rows[:max_scan]):
        roles = {}
        for j, c in enumerate(row):
            r = _role(c)
            if r and r not in roles:
                roles[r] = j
        if "field" in roles or "customer" in roles:
            if sum(bool(c) for c in row) >= 2:
                return i, roles
            if fallback[0] is None:
                fallback = (i, roles)
    return fallback


# ---------------------------------------------------------------- sheet parsers

def _builder():
    return {}  # customer name -> {"fields": [(name, type, desc, group, example)], "aliases": set()}


def _add(acc, customer, name, ftype="", desc="", group="", example="", alias=""):
    customer, name = clean_text(customer), clean_text(name)
    if not customer or not name:
        return
    c = acc.setdefault(customer, {"fields": [], "aliases": []})
    if all(slug(f[0]) != slug(name) or slug(f[3]) != slug(group) for f in c["fields"]):
        c["fields"].append((name, ftype, desc, group, example))
    for a in re.split(r"[;,\n]", alias or ""):
        if clean_text(a) and clean_text(a) not in c["aliases"]:
            c["aliases"].append(clean_text(a))


def _parse_sheet(sheet_name: str, rows: list[list[str]], acc) -> str:
    h, roles = _find_header(rows)
    if h is None:
        return "skipped (no header with a customer/field column)"
    header, body = rows[h], rows[h + 1:]
    width = len(header)
    known = set(roles.values())
    others = [j for j in range(width) if j not in known and header[j] and slug(header[j]) not in _IGNORE]

    def get(row, role):
        j = roles.get(role)
        return row[j] if j is not None and j < len(row) else ""

    if "customer" in roles and "field" in roles:  # long
        cur = ""
        for row in body:
            cur = get(row, "customer") or cur
            _add(acc, cur, get(row, "field"), get(row, "type"), get(row, "description"), get(row, "group"),
                 get(row, "example"), get(row, "alias"))
        return "long"

    if "field" in roles:
        marks = [_mark(row[j]) for row in body for j in others if j < len(row) and get(row, "field")]
        if others and marks and sum(m is not None for m in marks) >= 0.8 * len(marks):  # matrix
            for row in body:
                for j in others:
                    if j < len(row) and _mark(row[j]):
                        _add(acc, header[j], get(row, "field"), get(row, "type"), get(row, "description"),
                             get(row, "group"), get(row, "example"))
            return "matrix"
        # one sheet per customer; "Khách hàng: X" above the header wins over the sheet name
        customer = sheet_name
        for row in rows[:h]:
            for c in row:
                m = re.match(r"^\s*(kh[aá]ch\s*h[aà]ng|customer)\s*[:\-]\s*(.+)$", c, re.I)
                if m:
                    customer = m.group(2)
        for row in body:
            _add(acc, customer, get(row, "field"), get(row, "type"), get(row, "description"), get(row, "group"),
                 get(row, "example"), get(row, "alias"))
        return "sheet"

    # wide: one row per customer
    for row in body:
        cust = get(row, "customer")
        if not cust:
            continue
        for j in others:
            v = row[j] if j < len(row) else ""
            m = _mark(v)
            if m is False:
                continue
            _add(acc, cust, header[j], example="" if m else v, alias=get(row, "alias"))
    return "wide"


def _finalize(acc) -> dict[str, CustomerSpec]:
    specs = {}
    for cust, c in acc.items():
        fields, tables = [], {}
        for name, ftype, desc, group, example in c["fields"]:
            f = Field(name=name, type=parse_type(ftype, name), description=desc, example=example)
            gslug = slug(group)
            if gslug and (f.type == "table" or any(k in gslug for k in _TABLE_GROUP_HINTS)):
                if gslug not in tables:
                    tables[gslug] = Field(name=clean_text(group), type="table")
                    fields.append(tables[gslug])
                if f.type == "table":  # the row describes the table itself
                    tables[gslug].description = tables[gslug].description or desc
                    continue
                tables[gslug].columns.append(f)
            else:
                fields.append(f)
        specs[cust] = CustomerSpec(name=cust, fields=fields, aliases=c["aliases"])
    return specs


# ---------------------------------------------------------------- public API

def load_specs(path: str | Path, verbose: bool = False) -> dict[str, CustomerSpec]:
    """Load customer specs from .xlsx/.xls/.csv/.yaml/.json; a sibling `<stem>.yaml` is merged on top."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml", ".json"):
        return _load_structured(path)
    import pandas as pd

    if suffix == ".csv":
        sheets = {path.stem: pd.read_csv(path, header=None, dtype=str)}
    else:
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
    acc = _builder()
    for name, df in sheets.items():
        rows = [_cells(r) for r in df.itertuples(index=False)]
        layout = _parse_sheet(str(name), rows, acc)
        if verbose:
            print(f"[sheet '{name}'] layout: {layout}", file=sys.stderr)
    specs = _finalize(acc)
    for side in (path.with_suffix(".yaml"), path.with_suffix(".yml")):
        if side.exists():
            specs.update(_load_structured(side))
    return specs


def _load_structured(path: Path) -> dict[str, CustomerSpec]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    items = data.get("customers", data) if isinstance(data, dict) else data
    if isinstance(items, dict):  # {customer: [fields]} or {customer: {fields: [...]}}
        items = [{"name": k, **(v if isinstance(v, dict) else {"fields": v})} for k, v in items.items()]
    specs = [CustomerSpec.from_dict(d) for d in items]
    return {s.name: s for s in specs}


def dump_specs(specs: dict[str, CustomerSpec]) -> str:
    import yaml

    return yaml.safe_dump({"customers": [s.to_dict() for s in specs.values()]}, allow_unicode=True, sort_keys=False)


def find_spec_file(folder: str | Path) -> Path | None:
    """The Excel file in `folder` (its sibling YAML is merged by load_specs), else a standalone YAML/JSON."""
    folder = Path(folder)
    if not folder.exists():
        return None
    for patterns in (("*.xlsx", "*.xlsm", "*.xls"), ("*.yaml", "*.yml")):
        files = sorted(p for pat in patterns for p in folder.rglob(pat) if not p.name.startswith("~$"))
        if files:
            return files[0]
    return None


def default_spec_path() -> Path | None:
    return DEFAULT_SPEC if DEFAULT_SPEC.exists() else find_spec_file(PO_DIR)


def find_dms_file(folder: str | Path = PO_DIR) -> Path | None:
    """The store-code Excel (DS_SIEU_THI_VA_MA_DMS_*.xlsx) in `folder`, else any Excel file there."""
    folder = Path(folder)
    if not folder.exists():
        return None
    files = sorted(p for pat in ("*.xlsx", "*.xlsm") for p in folder.rglob(pat) if not p.name.startswith("~$"))
    return next((p for p in files if "dms" in slug(p.stem)), files[0] if files else None)


# ---------------------------------------------------------------- customer detection

def detect_customer(specs: dict[str, CustomerSpec], pdf_path: str | Path | None = None,
                    text: str = "") -> tuple[str | None, str]:
    """Pick the customer for a PO from its path (file/folder names) then its text layer.

    Returns (customer name or None, explanation). Short keys (< 4 chars) must match a whole word.
    """
    path_words = words(str(pdf_path)) if pdf_path else []
    path_slug = slug(str(pdf_path)) if pdf_path else ""
    text_words = words(text)
    text_slug = slug(text)
    best, best_score, why = None, 0.0, "không nhận diện được khách hàng"
    for name, spec in specs.items():
        for key in [name, *spec.aliases]:
            k = slug(key)
            if not k:
                continue
            kw = words(key)

            def hit(ws, s):
                if len(k) < 4:
                    return k in ws
                return k in s or (len(kw) > 1 and _contains_seq(ws, kw))

            score = 0.0
            if hit(path_words, path_slug):
                score = 2.0 + len(k) / 100
                src = "tên file/thư mục"
            elif hit(text_words, text_slug):
                score = 1.0 + len(k) / 100
                src = "nội dung PDF"
            if score > best_score:
                best, best_score, why = name, score, f"khớp '{key}' trong {src}"
    return best, why


def _contains_seq(hay: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Print the per-customer field specs understood from the Excel file")
    ap.add_argument("path", nargs="?", default=None, help="Excel/YAML spec (default: config/po_customers.yaml)")
    ap.add_argument("--yaml-out", help="write the specs as YAML (edit it, then put it next to the Excel to override)")
    args = ap.parse_args(argv)
    path = args.path or default_spec_path()
    if not path:
        sys.exit("No spec file given and no config/po_customers.yaml")
    specs = load_specs(path, verbose=True)
    text = dump_specs(specs)
    print(text)
    print(f"{len(specs)} customers", file=sys.stderr)
    if args.yaml_out:
        Path(args.yaml_out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
