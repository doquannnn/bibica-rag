import json

import pandas as pd

from bibica_rag.po import ui


def test_export_contains_only_the_extracted_information():
    results = [
        {"file": "a.pdf", "customer": "COOP", "customer_reason": "", "missing": [], "elapsed_s": 1, "error": "",
         "warnings": [], "dms": {"Mã DMS": "9352", "ghi_chu": ""}, "raw_responses": ["..."],
         "fields": {"Số PO": "1", "Chi tiết hàng": [{"Mã SP": "A", "SL đặt": 2}]}},
        {"file": "b.pdf", "customer": None, "customer_reason": "", "missing": [], "elapsed_s": 1, "error": "x",
         "fields": {}},
    ]
    xl, js = ui.export(results)
    sheets = pd.read_excel(xl, sheet_name=None, dtype=str)
    assert set(sheets) == {"COOP", "COOP - Chi tiết hàng"}
    assert sheets["COOP"].iloc[0].to_dict() == {"File": "a.pdf", "Số PO": "1", "Mã DMS": "9352"}
    assert sheets["COOP - Chi tiết hàng"].iloc[0].to_dict() == {"File": "a.pdf", "Số PO": "1", "Mã SP": "A",
                                                                "SL đặt": "2"}
    assert json.loads(open(js, encoding="utf-8").read()) == [
        {"file": "a.pdf", "Khách hàng": "COOP", "Số PO": "1", "Chi tiết hàng": [{"Mã SP": "A", "SL đặt": 2}],
         "Mã DMS": "9352"}]


def test_tab_builds():
    import gradio as gr

    with gr.Blocks() as demo:
        ui.build_po_tab()
    labels = {getattr(b, "label", None) for b in demo.blocks.values()}
    assert {"File PO (PDF)", "Chi tiết hàng", "Tải kết quả (Excel / JSON)"} <= labels
