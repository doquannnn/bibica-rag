"""Checks on the real POs in 'PO Extraction/' against the hand-checked GT in data/po_gt/.

- test_real_po_preprocessing / test_gt_is_grounded / test_dms_codes: no model needed
- test_real_po_accuracy: calls the VLM (default: local Qwen3.5-4B, `scripts/serve.sh 4B`; see bibica_rag/po/vlm.py):
      BIBICA_LIVE=1 uv run pytest tests/test_po_live.py -s
  Gates: customer detection 100%, exactly the customer's fields, header accuracy and line F1 >=
  BIBICA_PO_MIN_ACC (default 0.9).
"""
import json
import os
from pathlib import Path

import pytest

from bibica_rag.po.dms import load_dms_mapping
from bibica_rag.po.evaluate import report_md, score_dir
from bibica_rag.po.extract import extract_po, iter_pdfs, select_pages, text_is_usable
from bibica_rag.po.pdf import load_pdf
from bibica_rag.po.dms import lookup
from bibica_rag.po.spec import DEFAULT_SPEC, PO_DIR, default_spec_path, detect_customer, find_dms_file, load_specs
from bibica_rag.po.validate import validate

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


def test_gt_is_grounded():
    """The GT itself passes the self-checks: codes and quantities are printed in the PDF text layer (digital PDFs),
    EAN-13 check digits are valid, and line quantities add up to the total printed on the PO."""
    specs = load_specs(default_spec_path())
    printed_total = {"BHX ok": 420, "BigC ok": 197, "Coop ok": 639, "Lotte ok": 16, "Mega": 484, "satra 3": 26,
                     "VIn": 1635}
    for stem, (pdf, gt) in _cases().items():
        spec = specs[gt["customer"]]
        pages = load_pdf(pdf, max_pages=30)
        usable = text_is_usable("\n".join(pages.texts))
        text = "\n".join(pages.texts[i] for i in select_pages(spec, pages)) if usable else ""
        assert validate(spec, gt["fields"], text) == [], stem
        if stem in printed_total:
            assert sum(r["SL đặt"] for r in gt["fields"]["Chi tiết hàng"]) == printed_total[stem], stem


def test_dms_codes():
    """Store codes of the samples looked up in the Excel; unknown ones must be reported, never guessed."""
    specs = load_specs(default_spec_path())
    mapping = load_dms_mapping(find_dms_file())
    got = {stem: lookup(mapping, specs[gt["customer"]].dms_sheet, gt["fields"]["Mã KH"])["Mã DMS"]
           for stem, (_, gt) in _cases().items()}
    assert got == {"Aeon ok": None, "BHX ok": None, "BigC ok": "12032", "Coop ok": None, "EMART ok": "7120",
                   "Lotte ok": None, "Mega": "7683", "satra 3": "7537", "VIn": None}


@pytest.mark.skipif(os.environ.get("BIBICA_LIVE") != "1", reason="set BIBICA_LIVE=1 (calls the VLM)")
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
    min_acc = float(os.environ.get("BIBICA_PO_MIN_ACC", "0.9"))
    assert rep["header_acc_strict"] >= min_acc and rep["line_f1"] >= min_acc


# Dates and quantities exactly as printed on each sample PO (what a VLM following the prompt should copy)
PRINTED_DATE = {"Aeon ok": "14/11/2023", "BHX ok": "07/10/2023", "BigC ok": "09/11/23", "Coop ok": "6/11/23",
                "EMART ok": "17.11.2023", "Lotte ok": "2023-11-16", "Mega": "17.11.2023", "satra 3": "11/28/2023",
                "VIn": "23.11.2023"}
PRINTED_QTY = {"satra 3": lambda q: f"{q:.3f}".replace(".", ","), "Mega": lambda q: f"{q:.2f}".replace(".", ","),
               "BHX ok": lambda q: f"{q:,.2f}", "VIn": lambda q: f"{q:,.0f}"}


def test_pipeline_on_real_pdfs_with_scripted_vlm(tmp_path, fake_client):
    """End to end on the real PDFs with a VLM stand-in that copies values as printed: page selection (BigC),
    per-page line merging, +5 days, month-first dates (Satra), decimal commas, DMS lookup and the evaluator must
    then give 100%. Measures the plumbing, not a model."""
    from types import SimpleNamespace

    specs = load_specs(default_spec_path())
    mapping = load_dms_mapping(find_dms_file())
    for stem, (pdf, gt) in _cases().items():
        spec = specs[gt["customer"]]
        rows = gt["fields"]["Chi tiết hàng"]
        fmt = PRINTED_QTY.get(stem, lambda q: q)
        printed = [{**r, "SL đặt": fmt(r["SL đặt"])} for r in rows]

        def reply(**kw):
            prompt = kw["messages"][1]["content"][-1]["text"]
            page = prompt.split("<pdf_text>")[1] if "<pdf_text>" in prompt else ""
            lines = [r for r in printed if r["Mã SP"] in page] if page else printed
            if "Chỉ lấy các dòng hàng" in prompt:
                return {"Chi tiết hàng": lines}
            head = {k: v for k, v in gt["fields"].items() if k != "Chi tiết hàng"}
            return {**head, "Ngày đề nghị giao hàng": PRINTED_DATE[stem], "Chi tiết hàng": lines}

        client = fake_client()
        client.chat = SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps(reply(**kw), ensure_ascii=False)))])))
        r = extract_po(pdf, specs, client=client, model="scripted", dms_mapping=mapping, max_pages=30)
        assert not r.error, f"{stem}: {r.error}"
        if stem == "BigC ok":
            assert r.pages_used == [1, 2]
        (tmp_path / f"{stem}.json").write_text(json.dumps(r.to_dict(), ensure_ascii=False), encoding="utf-8")
    rep = score_dir(tmp_path, GT_DIR, specs, only_checked=True)
    assert rep["order_exact"] == 1, report_md(rep)


def test_prompt_examples_do_not_leak_the_ground_truth():
    """Examples in the field descriptions come from the Excel screenshots (2021 POs), not from the evaluated samples;
    otherwise the accuracy would be inflated. Allowed: fixed codes the Excel itself gives for the chain."""
    from_excel = {"8935132500710", "VD-00001986", "8934609203093"}
    spec_text = DEFAULT_SPEC.read_text(encoding="utf-8")
    leaked = set()
    for _, gt in _cases().values():
        for k, v in gt["fields"].items():
            values = [r["Mã SP"] for r in v] if isinstance(v, list) else [v]
            leaked |= {str(x) for x in values if x and len(str(x)) >= 4 and str(x) in spec_text}
    assert leaked <= from_excel, leaked - from_excel
