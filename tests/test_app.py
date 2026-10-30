import pandas as pd

from bibica_rag import app


def test_app_builds_with_both_tabs():
    blocks = app.build_app()
    labels = {getattr(b, "label", None) for b in blocks.blocks.values()}
    assert {"📑 Trích xuất PO", "💬 Hỏi đáp tài liệu nội bộ"} <= labels


def test_export_one_sheet_per_customer_and_table():
    results = [
        {"file": "a.pdf", "customer": "Co.opmart", "customer_reason": "", "missing": [], "elapsed_s": 1, "error": "",
         "fields": {"Số PO": "1", "Hàng": [{"Mã": "A", "SL": 2}]}},
        {"file": "b.pdf", "customer": "Lotte", "customer_reason": "", "missing": [], "elapsed_s": 1, "error": "",
         "fields": {"Số đơn hàng": "L-1"}},
    ]
    xl, js = app.export(results)
    sheets = pd.read_excel(xl, sheet_name=None)
    assert set(sheets) == {"Co.opmart", "Co.opmart - Hàng", "Lotte"}
    assert list(sheets["Co.opmart"].columns) == ["File", "Số PO"]
    assert list(sheets["Lotte"].columns) == ["File", "Số đơn hàng"]
    assert sheets["Co.opmart - Hàng"].iloc[0].to_dict() == {"File": "a.pdf", "Mã": "A", "SL": 2}
