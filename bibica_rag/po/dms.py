"""Store code on the PO -> DMS customer code, read from the Excel file in `PO Extraction/`
(DS_SIEU_THI_VA_MA_DMS_*.xlsx): one sheet per retailer with the columns
| TÊN SIÊU THỊ | MÃ TRÊN PO PDF | MÃ DMS | TÊN SIÊU THỊ DMS |  and a note like "Đơn vị tính Thùng".
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

from bibica_rag.po.normalize import clean_text, slug

_CODE_HEADERS = ("matrenpopdf", "mast", "masieuthi", "macuahang")
_DMS_HEADERS = ("madms",)


@dataclass
class StoreRow:
    store_name: str
    po_code: str
    dms_code: str
    dms_name: str


@dataclass
class DmsSheet:
    name: str
    rows: list[StoreRow] = field(default_factory=list)
    unit: str = ""  # "Đơn vị tính ..." note of the sheet


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return clean_text(v)


def code_key(code) -> str:
    """'01003' == '1003', 'M1500806' == 'm1500806', '7211 - BHX_...' keeps only what is compared."""
    return slug(code).lstrip("0")


def load_dms_mapping(path: str | Path) -> dict[str, DmsSheet]:
    from openpyxl import load_workbook

    wb = load_workbook(path, data_only=True, read_only=True)
    out = {}
    for ws in wb.worksheets:
        rows = [[_cell(v) for v in r] for r in ws.iter_rows(values_only=True)]
        sheet = DmsSheet(name=ws.title)
        for r in rows:
            for c in r:
                m = re.match(r"(?i)^đơn vị tính\s*(.*)$", c)
                if m and not sheet.unit:
                    sheet.unit = m.group(1).strip()
        header = None
        for i, r in enumerate(rows):
            slugs = [slug(c) for c in r]
            code_col = next((j for j, s in enumerate(slugs) if s in _CODE_HEADERS), None)
            dms_col = next((j for j, s in enumerate(slugs) if s in _DMS_HEADERS), None)
            if code_col is not None and dms_col is not None:
                name_col = next((j for j, s in enumerate(slugs) if s.startswith("tensieuthi") and "dms" not in s
                                 or s == "tenst"), None)
                dms_name_col = next((j for j, s in enumerate(slugs) if s == "tensieuthidms"), None)
                header = (i, name_col, code_col, dms_col, dms_name_col)
                break
        if header:
            i, name_col, code_col, dms_col, dms_name_col = header
            get = lambda r, j: r[j] if j is not None and j < len(r) else ""  # noqa: E731
            for r in rows[i + 1:]:
                if get(r, code_col) or get(r, dms_col):
                    sheet.rows.append(StoreRow(get(r, name_col), get(r, code_col), get(r, dms_col),
                                               get(r, dms_name_col)))
        if sheet.rows or sheet.unit:
            out[ws.title] = sheet
    return out


def lookup(mapping: dict[str, DmsSheet], sheet_name: str, po_code) -> dict:
    """{'Mã DMS', 'Tên siêu thị DMS', 'Đơn vị tính DMS', 'ghi_chu'} for a store code read from the PO."""
    sheet = next((s for n, s in mapping.items() if slug(n) == slug(sheet_name)), None) if sheet_name else None
    res = {"Mã DMS": None, "Tên siêu thị DMS": None, "Đơn vị tính DMS": sheet.unit if sheet else None,
           "ghi_chu": ""}
    if sheet is None:
        res["ghi_chu"] = f"không có sheet '{sheet_name}' trong file mã DMS"
        return res
    if po_code in (None, ""):
        res["ghi_chu"] = "không đọc được mã KH trên PO"
        return res
    key = code_key(po_code)
    for row in sheet.rows:
        if row.po_code and code_key(row.po_code) == key:
            res["Mã DMS"], res["Tên siêu thị DMS"] = row.dms_code or None, row.dms_name or row.store_name or None
            return res
    # codes made of words (E-Mart "366 PHAN") are matched inside the longer text read from the PO
    # ("EMART GO VAP 366 PHAN VAN TRI,P5,..."); purely numeric codes must match exactly
    for row in sheet.rows:
        if row.po_code and re.search(r"[A-Za-z]", row.po_code) and " " in row.po_code.strip() \
                and re.search(rf"(?<![0-9a-z]){re.escape(' '.join(row.po_code.lower().split()))}(?![0-9])",
                              " ".join(str(po_code).lower().split())):
            res["Mã DMS"], res["Tên siêu thị DMS"] = row.dms_code or None, row.dms_name or row.store_name or None
            return res
    res["ghi_chu"] = f"mã KH '{po_code}' chưa có trong sheet '{sheet.name}' — cần bổ sung vào file mã DMS"
    return res
