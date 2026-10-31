import json

from bibica_rag.po.evaluate import compare_table, init_gt, report_md, score_dir
from bibica_rag.po.spec import CustomerSpec, Field

ITEMS = Field("Hàng", "table", columns=[Field("Mã"), Field("SL", "number")])
SPECS = {"Lotte": CustomerSpec("Lotte", [Field("Số đơn hàng"), Field("Ngày giao", "date"), ITEMS])}


def _write(d, name, obj):
    d.mkdir(exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def test_lines_are_matched_by_code_not_position():
    gt = [{"Mã": "A", "SL": 1}, {"Mã": "B", "SL": 2}, {"Mã": "C", "SL": 3}, {"Mã": "D", "SL": 4}]
    pred = [{"Mã": "A", "SL": 1}, {"Mã": "C", "SL": 3}, {"Mã": "D", "SL": 5}, {"Mã": "X", "SL": 9}]  # B skipped
    t = compare_table(ITEMS, pred, gt)
    assert (t["n_pred"], t["n_gt"], t["matched"], t["exact"]) == (4, 4, 3, 2)  # C still counts as correct
    assert t["missed"] == ["B"] and t["extra"] == ["X"]
    assert t["col_hits"] == {"SL": 2} and t["wrong"] == [{"pred": {"Mã": "D", "SL": 5}, "gt": {"Mã": "D", "SL": 4}}]
    # duplicated codes are matched one to one
    t = compare_table(ITEMS, [{"Mã": "A", "SL": 1}] * 3, [{"Mã": "A", "SL": 1}] * 2)
    assert (t["matched"], t["extra"]) == (2, ["A"])


def test_score_dir_metrics_and_init_gt(tmp_path):
    pred, gt = tmp_path / "pred", tmp_path / "gt"
    _write(pred, "a", {"customer": "Lotte", "error": "", "fields": {
        "Số đơn hàng": "l-77 ", "Ngày giao": "2025-03-05", "Hàng": [{"Mã": "A", "SL": 2}, {"Mã": "B", "SL": 3}]}})
    _write(pred, "b", {"customer": "Lotte", "error": "", "warnings": ["Dòng 1 (Z): mã không có trong lớp chữ"],
                       "fields": {"Số đơn hàng": "Kho Bình Dương", "Ngày giao": None, "Hàng": [{"Mã": "Z", "SL": 1}]}})
    assert init_gt(pred, gt) == 2 and init_gt(pred, gt) == 0  # never overwrites corrected GT
    _write(gt, "a", {"customer": "Lotte", "checked": True, "fields": {
        "Số đơn hàng": "L-77", "Ngày giao": "05/03/2025", "Hàng": [{"Mã": "A", "SL": "2"}, {"Mã": "B", "SL": 3}]}})
    _write(gt, "b", {"customer": "Lotte", "checked": True, "fields": {
        "Số đơn hàng": "Kho Binh Duong", "Ngày giao": "06/03/2025",
        "Hàng": [{"Mã": "C", "SL": 1}, {"Mã": "Z", "SL": 2}]}})
    rep = score_dir(pred, gt, SPECS)
    assert rep["n_files"] == 2 and rep["customer_acc"] == 1 and rep["spec_conformity"] == 1
    assert rep["order_exact"] == 0.5
    assert rep["header_acc_strict"] == 2 / 4 and rep["header_acc_loose"] == 3 / 4  # loose ignores accents
    assert rep["line_precision"] == 3 / 3 and rep["line_recall"] == 3 / 4
    assert rep["line_exact_rate"] == 2 / 4 and rep["column_acc_on_matched"] == {"SL": 0.6667}
    md = report_md(rep)
    assert "| b | Ngày giao | null |" in md and "missed lines" in md and "mã không có trong lớp chữ" in md


def test_missing_prediction_counts_as_wrong(tmp_path):
    gt = tmp_path / "gt"
    _write(gt, "a", {"customer": "Lotte", "checked": True, "fields": {
        "Số đơn hàng": "1", "Ngày giao": None, "Hàng": [{"Mã": "A", "SL": 1}]}})
    rep = score_dir(tmp_path / "none", gt, SPECS)
    assert rep["order_exact"] == 0 and rep["line_recall"] == 0 and rep["files"][0]["error"] == "no prediction"
