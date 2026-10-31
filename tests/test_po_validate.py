from bibica_rag.po.normalize import normalize_value, parse_date, parse_number
from bibica_rag.po.spec import CustomerSpec, Field
from bibica_rag.po.validate import ean13_ok, validate
from bibica_rag.po.vlm import GEMINI_BASE_URL, backend_from_env, call_with_retry

SPEC = CustomerSpec("X", [Field("Số PO"), Field("Ngày", "date"),
                          Field("Hàng", "table", columns=[Field("Mã SP"), Field("SL đặt", "number")])])
TEXT = """PO No: 4500929220   Delivery Date: 17.11.2023
1003086 8934609471355 KEO DEO TUI AO DAU 500G BOX 12 3 450.000
1003087 8934609103195 KEO DEO TUI AO DUONG 500G BOX 12 2 450.000"""


def test_ean13():
    assert ean13_ok("8934609471355") and ean13_ok("8934609103195")
    assert not ean13_ok("8934609471354")  # one digit misread
    assert ean13_ok("3003997-0")  # not an EAN: not checked


def test_validate_clean_extraction_has_no_warning():
    fields = {"Số PO": "4500929220", "Ngày": "2023-11-17",
              "Hàng": [{"Mã SP": "8934609471355", "SL đặt": 3}, {"Mã SP": "8934609103195", "SL đặt": 2}]}
    assert validate(SPEC, fields, TEXT) == []


def test_validate_flags_misreads():
    fields = {"Số PO": "4500929221", "Ngày": None,
              "Hàng": [{"Mã SP": "8934609471354", "SL đặt": 3}, {"Mã SP": "8934609103195", "SL đặt": 20},
                       {"Mã SP": "8934609103195", "SL đặt": 0}]}
    w = "\n".join(validate(SPEC, fields, TEXT, {"ghi_chu": "mã KH '1291' chưa có"}))
    assert "'Số PO' = '4500929221' không tìm thấy" in w and "Thiếu 'Ngày'" in w
    assert "EAN-13" in w and "Dòng 1 (8934609471354): mã không có" in w
    assert "số lượng 20 không thấy" in w and "trùng mã với dòng 2" in w and "không hợp lệ (0)" in w
    assert "DMS: mã KH '1291'" in w
    # scanned PDF (no text layer): only the mechanical checks
    assert validate(SPEC, {**fields, "Ngày": "2023-01-01"}, "") and \
        not any("lớp chữ" in x for x in validate(SPEC, fields, ""))


def test_satra_style_numbers_and_dates():
    assert parse_number("1,000", decimal_comma=True) == 1 and parse_number("24,000", decimal_comma=True) == 24
    assert parse_number("1,000") == 1000  # default: thousands separator
    assert normalize_value("60,00", "number", decimal_comma=True) == 60
    assert normalize_value(1, "number", decimal_comma=True) == 1  # already a number: untouched
    assert parse_date("11/28/2023", "mdy").isoformat() == "2023-11-28"
    assert parse_date("11/12/2023", "mdy").isoformat() == "2023-11-12"
    assert parse_date("11/12/2023").isoformat() == "2023-12-11"
    assert normalize_value("2023-11-28", "date", date_order="mdy") == "2023-11-28"  # ISO stays ISO


def test_backend_selection(monkeypatch):
    for k in ("BIBICA_PO_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY", "BIBICA_PO_MODEL", "BIBICA_PO_BASE_URL",
              "BIBICA_PO_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    assert backend_from_env().provider == "local"
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    b = backend_from_env()
    assert (b.provider, b.base_url, b.model, b.api_key, b.local) == ("gemini", GEMINI_BASE_URL, "gemini-3.5-flash",
                                                                     "k", False)
    monkeypatch.setenv("BIBICA_PO_MODEL", "gemini-3.8-flash")
    assert backend_from_env().model == "gemini-3.8-flash"
    assert backend_from_env("local").local


def test_retry_on_rate_limit(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)

    class RateLimited(Exception):
        status_code = 429

    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise RateLimited()
        return "ok"

    assert call_with_retry(fn) == "ok" and len(calls) == 3

    class Bad(Exception):
        status_code = 400

    try:
        call_with_retry(lambda: (_ for _ in ()).throw(Bad()))
        raise AssertionError
    except Bad:
        pass


def test_daily_quota_is_not_retried(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)

    class Quota(Exception):
        status_code = 429

        def __str__(self):
            return "Generate content requests per day limit exceeded on Free Tier"

    calls = []

    def fn():
        calls.append(1)
        raise Quota()

    try:
        call_with_retry(fn)
    except Quota:
        pass
    assert len(calls) == 1
