"""Text / number / date normalization shared by the extractor (post-processing) and the evaluator."""
import re
import unicodedata
from datetime import date


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def slug(s: str) -> str:
    """Lowercase, accent-free, alphanumerics only: 'Công ty ABC-01' -> 'congtyabc01'."""
    return re.sub(r"[^a-z0-9]", "", strip_accents(str(s)).lower())


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", strip_accents(str(s)).lower())


def clean_text(s) -> str:
    return re.sub(r"\s+", " ", nfc(str(s))).strip()


# ---------------------------------------------------------------- numbers

_NUM_CHARS = re.compile(r"[^0-9,.\-]")


def parse_number(v, decimal_comma: bool = False) -> float | None:
    """Parse '1.234.567,5' (VN), '1,234,567.5' (EN), '12 500 VND', 1234 -> float. None if not a number.

    decimal_comma: the document always writes decimals with a comma (Satra '1,000' = 1, MM '60,00' = 60)."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    neg = s.startswith("(") and s.endswith(")")
    s = _NUM_CHARS.sub("", s.replace(" ", "").replace(" ", ""))
    if not re.search(r"\d", s):
        return None
    neg = neg or s.startswith("-")
    s = s.replace("-", "")
    if decimal_comma and "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." in s:
        # the right-most separator is the decimal one
        dec, thou = (",", ".") if s.rfind(",") > s.rfind(".") else (".", ",")
        s = s.replace(thou, "").replace(dec, ".")
    elif "," in s or "." in s:
        sep = "," if "," in s else "."
        parts = s.split(sep)
        # '1.234.567' / '1,234' -> thousands; '12,5' / '3.75' -> decimal
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and parts[0] not in ("", "0")):
            s = "".join(parts)
        else:
            s = s.replace(sep, ".")
    try:
        x = float(s)
    except ValueError:
        return None
    return -x if neg else x


def format_number(x: float) -> int | float:
    return int(x) if float(x).is_integer() else round(x, 6)


# ---------------------------------------------------------------- dates

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def parse_date(v, order: str = "dmy") -> date | None:
    """Parse common PO date formats; day-first for ambiguous numeric dates (Vietnam convention) unless
    order == "mdy" (e.g. Satra prints 11/28/2023)."""
    if v is None:
        return None
    if isinstance(v, date):
        return v if type(v) is date else v.date()
    s = strip_accents(str(v)).lower().strip()
    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)  # ISO
    if m:
        y, mo, d = map(int, m.groups())
        return _mk(y, mo, d)
    m = re.search(r"ngay\s*(\d{1,2})\s*thang\s*(\d{1,2})\s*nam\s*(\d{4})", s)  # 'ngày 05 tháng 3 năm 2025'
    if m:
        d, mo, y = map(int, m.groups())
        return _mk(y, mo, d)
    m = re.search(r"(\d{1,2})[-/. ]([a-z]{3})[a-z]*[-/., ]+(\d{2,4})", s)  # '05-Mar-2025'
    if m and m.group(2) in _MONTHS:
        d, y = int(m.group(1)), int(m.group(3))
        return _mk(y + 2000 if y < 100 else y, _MONTHS[m.group(2)], d)
    m = re.search(r"([a-z]{3})[a-z]*\s+(\d{1,2}),?\s+(\d{4})", s)  # 'March 5, 2025'
    if m and m.group(1) in _MONTHS:
        return _mk(int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2)))
    m = re.search(r"(\d{1,2})(?:[-/.]|\s+)(\d{1,2})(?:[-/.]|\s+)(\d{4}|\d{2}\b)", s)  # dd/mm/yyyy, 'dd mm yyyy'
    if m:
        d, mo, y = map(int, m.groups())
        if order == "mdy":
            d, mo = mo, d
        if mo > 12 >= d:  # clearly the other order
            d, mo = mo, d
        return _mk(y + 2000 if y < 100 else y, mo, d)
    return None


def _mk(y, mo, d):
    try:
        return date(y, mo, d)
    except ValueError:
        return None


# ---------------------------------------------------------------- values by type

def normalize_value(v, ftype: str, date_order: str = "dmy", decimal_comma: bool = False):
    """Canonical form used both for the final output and for comparison in the evaluator."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in ("", "null", "none", "n/a", "na", "-", "không có"):
        return None
    if ftype == "number":
        x = parse_number(v, decimal_comma=decimal_comma and isinstance(v, str))
        return format_number(x) if x is not None else clean_text(v)
    if ftype == "date":
        d = parse_date(v, date_order)
        return d.isoformat() if d else clean_text(v)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return format_number(float(v))
    return clean_text(v)
