"""Checks on the real POs in 'PO Extraction/' against the hand-checked GT in data/po_gt/ (both kept local).

- test_real_po_preprocessing: no model needed — customer detection, text-layer check, page selection, GT/spec match
- test_real_po_accuracy: needs a running llama-server with the Qwen VLM:
      BIBICA_LIVE=1 uv run pytest tests/test_po_live.py -s
  Env: BIBICA_PO_MIN_ACC (default 0.9) minimum strict field accuracy over GT files marked "checked": true.
"""
import json
import os
from pathlib import Path

import pytest

from bibica_rag.po.dms import load_dms_mapping
from bibica_rag.po.evaluate import report_md, score_dir
from bibica_rag.po.extract import extract_po, iter_pdfs, select_pages, text_is_usable
from bibica_rag.po.pdf import load_pdf
from bibica_rag.po.spec import PO_DIR, default_spec_path, detect_customer, find_dms_file, load_specs

ROOT = Path(__file__).resolve().parents[1]
GT_DIR = ROOT / "data" / "po_gt"


def _cases():
    gts = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(GT_DIR.glob("*.json"))}
    gts = {k: v for k, v in gts.items() if v.get("checked")}
    pdfs = {p.stem: p for p in iter_pdfs(PO_DIR)} if PO_DIR.exists() else {}
    cases = {k: (pdfs[k], v) for k, v in gts.items() if k in pdfs}
    if not cases:
        pytest.skip("needs PDFs in 'PO Extraction/' and checked GT in data/po_gt/ (see bibica_rag/po/evaluate.py)")
    return cases


def test_real_po_preprocessing():
    specs = load_specs(default_spec_path())
    for stem, (pdf, gt) in _cases().items():
        spec = specs[gt["customer"]]
        assert list(gt["fields"]) == spec.field_names, f"{stem}: GT keys differ from the spec"
        pages = load_pdf(pdf, max_pages=30)
        text = "\n".join(pages.texts)
        usable = text_is_usable(text)
        assert detect_customer(specs, pdf, text if usable else "")[0] == gt["customer"], stem
        if usable and spec.skip_pages_containing:
            # pages sent to the VLM hold every GT line item code, and only those pages are kept
            kept = "\n".join(pages.texts[i] for i in select_pages(spec, pages))
            for row in gt["fields"]["Chi tiết hàng"]:
                assert row["Mã SP"] in kept, f"{stem}: {row['Mã SP']} not on a kept page"
            assert len(select_pages(spec, pages)) < pages.n_pages


@pytest.mark.skipif(os.environ.get("BIBICA_LIVE") != "1", reason="set BIBICA_LIVE=1 (needs llama-server)")
def test_real_po_accuracy(tmp_path):
    specs = load_specs(default_spec_path())
    dms_path = find_dms_file()
    mapping = load_dms_mapping(dms_path) if dms_path else None
    for stem, (pdf, gt) in _cases().items():
        r = extract_po(pdf, specs, dms_mapping=mapping)
        assert not r.error, f"{stem}: {r.error}"
        # exactly the customer's fields — nothing more, nothing less
        assert list(r.fields) == specs[r.customer].field_names
        (tmp_path / f"{stem}.json").write_text(json.dumps(r.to_dict(), ensure_ascii=False), encoding="utf-8")
    rep = score_dir(tmp_path, GT_DIR, specs, only_checked=True)
    print(report_md(rep))
    assert rep["customer_acc"] == 1
    assert rep["spec_conformity"] == 1
    assert rep["field_acc_strict"] >= float(os.environ.get("BIBICA_PO_MIN_ACC", "0.9"))
