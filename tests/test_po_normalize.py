import pytest

from bibica_rag.po.normalize import normalize_value, parse_date, parse_number


@pytest.mark.parametrize("raw,expected", [
    ("1.234.567", 1234567), ("1,234,567", 1234567), ("1.234.567,50", 1234567.5), ("1,234,567.50", 1234567.5),
    ("12,5", 12.5), ("3.75", 3.75), ("12 500 VNĐ", 12500), ("0.125", 0.125), (240, 240), ("(1.000)", -1000),
    ("abc", None), (None, None),
])
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("05/03/2025", "2025-03-05"), ("5-3-25", "2025-03-05"), ("2025-03-05", "2025-03-05"),
    ("ngày 05 tháng 3 năm 2025", "2025-03-05"), ("05-Mar-2025", "2025-03-05"), ("March 5, 2025", "2025-03-05"),
    ("03/25/2025", "2025-03-25"),  # month-first only when unambiguous
])
def test_parse_date(raw, expected):
    assert parse_date(raw).isoformat() == expected


def test_normalize_value():
    assert normalize_value("  Kho   Bình Dương ", "text") == "Kho Bình Dương"
    assert normalize_value("null", "text") is None
    assert normalize_value("1.500.000 đ", "number") == 1500000
    assert normalize_value("15/04/2025", "date") == "2025-04-15"
    assert normalize_value("không rõ", "date") == "không rõ"  # unparseable values are kept, not dropped
