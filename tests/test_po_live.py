"""Accuracy gate on the real POs — needs the local data, the hand-checked GT and a running llama-server.

    BIBICA_LIVE=1 uv run pytest tests/test_po_live.py -s
Env: BIBICA_PO_MIN_ACC (default 0.9) minimum strict field accuracy over GT files marked "checked": true.
"""
import json
import os
from pathlib import Path

import pytest

from bibica_rag.po.evaluate import report_md, score_dir
from bibica_rag.po.extract import PO_DIR, extract_po, iter_pdfs
from bibica_rag.po.spec import find_spec_file, load_specs

ROOT = Path(__file__).resolve().parents[1]
GT_DIR = ROOT / "data" / "po_gt"

pytestmark = pytest.mark.skipif(os.environ.get("BIBICA_LIVE") != "1", reason="set BIBICA_LIVE=1 to run")


def test_real_po_accuracy(tmp_path):
    spec_path = find_spec_file(PO_DIR)
    if not spec_path or not GT_DIR.exists():
        pytest.skip("needs 'PO Extraction/' (PDFs + Excel) and data/po_gt/ (see bibica_rag/po/evaluate.py)")
    specs = load_specs(spec_path)
    gts = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in GT_DIR.glob("*.json")}
    gts = {k: v for k, v in gts.items() if v.get("checked")}
    if not gts:
        pytest.skip('no GT file marked "checked": true')
    pdfs = {p.stem: p for p in iter_pdfs(PO_DIR)}
    for stem in gts:
        r = extract_po(pdfs[stem], specs)
        assert not r.error, f"{stem}: {r.error}"
        # exactly the customer's fields — nothing more, nothing less
        assert list(r.fields) == specs[r.customer].field_names
        (tmp_path / f"{stem}.json").write_text(json.dumps(r.to_dict(), ensure_ascii=False), encoding="utf-8")
    rep = score_dir(tmp_path, GT_DIR, specs, only_checked=True)
    print(report_md(rep))
    assert rep["customer_acc"] == 1
    assert rep["spec_conformity"] == 1
    assert rep["field_acc_strict"] >= float(os.environ.get("BIBICA_PO_MIN_ACC", "0.9"))
