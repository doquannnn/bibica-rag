import json

from bibica_rag.po.evaluate import init_gt, report_md, score_dir
from bibica_rag.po.spec import CustomerSpec, Field

SPECS = {"Lotte": CustomerSpec("Lotte", [
    Field("Số đơn hàng"), Field("Ngày giao", "date"), Field("Tổng tiền", "number"),
    Field("Hàng", "table", columns=[Field("Mã"), Field("SL", "number")])])}


def _write(d, name, obj):
    d.mkdir(exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def test_score_and_init_gt(tmp_path):
    pred, gt = tmp_path / "pred", tmp_path / "gt"
    _write(pred, "a", {"customer": "Lotte", "error": "", "fields": {
        "Số đơn hàng": "l-77 ", "Ngày giao": "2025-03-05", "Tổng tiền": 1500000,
        "Hàng": [{"Mã": "A", "SL": 2}, {"Mã": "B", "SL": 3}]}})
    _write(pred, "b", {"customer": "Lotte", "error": "", "fields": {
        "Số đơn hàng": "Kho Bình Dương", "Ngày giao": None, "Tổng tiền": 10, "Hàng": []}})
    assert init_gt(pred, gt) == 2 and init_gt(pred, gt) == 0  # never overwrites corrected GT
    # human corrections: a is right (different formatting), b has 2 wrong fields
    _write(gt, "a", {"customer": "Lotte", "checked": True, "fields": {
        "Số đơn hàng": "L-77", "Ngày giao": "05/03/2025", "Tổng tiền": "1.500.000",
        "Hàng": [{"Mã": "A", "SL": "2"}, {"Mã": "B", "SL": 3}]}})
    _write(gt, "b", {"customer": "Lotte", "checked": True, "fields": {
        "Số đơn hàng": "Kho Binh Duong", "Ngày giao": "06/03/2025", "Tổng tiền": 10, "Hàng": [{"Mã": "C", "SL": 1}]}})
    rep = score_dir(pred, gt, SPECS)
    assert rep["n_files"] == 2 and rep["customer_acc"] == 1 and rep["spec_conformity"] == 1
    assert rep["file_exact"] == 0.5
    assert rep["field_acc_strict"] == 5 / 8  # b: accents differ, date missing, table empty
    assert rep["field_acc_loose"] == 6 / 8  # loose ignores the missing accents
    md = report_md(rep)
    assert "| b | Ngày giao | null |" in md
