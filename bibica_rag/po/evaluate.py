"""Accuracy of the PO extractor against hand-checked ground truth.

Ground truth: one JSON per PDF in `data/po_gt/<pdf stem>.json` (not tracked — customer data):
    {"customer": "<name as in the Excel>", "fields": {"<field>": <value or null>, "<table field>": [{...}, ...]}}

Workflow
    # 1. run the extractor, then turn its output into GT templates and correct them by hand
    python -m bibica_rag.po.extract "PO Extraction/" --out results/po
    python -m bibica_rag.po.evaluate init-gt --pred results/po --gt data/po_gt
    # 2. after correcting data/po_gt/*.json (set "checked": true), score any later run
    python -m bibica_rag.po.evaluate score --pred results/po --gt data/po_gt

Metrics: customer detection accuracy, field accuracy (strict = case/space-insensitive equality after type
normalization; loose = accents/punctuation ignored too), file exact-match rate, and spec conformity
(output keys == the customer's field list, nothing extra, nothing missing).
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from bibica_rag.po.normalize import clean_text, normalize_value, slug
from bibica_rag.po.spec import CustomerSpec, Field, default_spec_path, load_specs

ROOT = Path(__file__).resolve().parents[2]


def values_equal(pred, gt, ftype: str, loose: bool = False) -> bool:
    p, g = normalize_value(pred, ftype), normalize_value(gt, ftype)
    if p is None or g is None:
        return p is None and g is None
    if isinstance(p, (int, float)) and isinstance(g, (int, float)):
        return math.isclose(p, g, rel_tol=1e-6, abs_tol=1e-6)
    if loose:
        return slug(p) == slug(g)
    return clean_text(p).casefold() == clean_text(g).casefold()


def compare_field(f: Field, pred, gt, loose: bool = False) -> tuple[bool, float]:
    """(field fully correct, fraction of correct cells — 1/0 for scalars)."""
    if f.type != "table":
        ok = values_equal(pred, gt, f.type, loose)
        return ok, float(ok)
    pred, gt = pred or [], gt or []
    if not pred and not gt:
        return True, 1.0
    cols = [c for c in f.columns] or [Field(name=k) for k in sorted({k for r in gt + pred for k in r})]
    n = max(len(pred), len(gt)) * len(cols)
    hits = 0
    for i in range(min(len(pred), len(gt))):
        for c in cols:
            hits += values_equal(pred[i].get(c.name), gt[i].get(c.name), c.type, loose)
    return hits == n and len(pred) == len(gt), hits / n if n else 1.0


def score_file(spec: CustomerSpec, pred: dict, gt: dict) -> dict:
    pf, gf = pred.get("fields", {}), gt.get("fields", {})
    rows = []
    for f in spec.fields:
        strict, cells = compare_field(f, pf.get(f.name), gf.get(f.name))
        loose, _ = compare_field(f, pf.get(f.name), gf.get(f.name), loose=True)
        rows.append({"field": f.name, "type": f.type, "strict": strict, "loose": loose, "cell_acc": cells,
                     "pred": pf.get(f.name), "gt": gf.get(f.name)})
    return {"customer_ok": pred.get("customer") == gt.get("customer"),
            "conform": list(pf) == spec.field_names, "fields": rows,
            "exact": all(r["strict"] for r in rows)}


def score_dir(pred_dir: Path, gt_dir: Path, specs: dict[str, CustomerSpec], only_checked: bool = False) -> dict:
    files, per_field, per_customer = [], defaultdict(lambda: [0, 0, 0]), defaultdict(lambda: [0, 0, 0])
    for gt_path in sorted(gt_dir.glob("*.json")):
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        if only_checked and not gt.get("checked"):
            continue
        pred_path = pred_dir / gt_path.name
        pred = json.loads(pred_path.read_text(encoding="utf-8")) if pred_path.exists() else {}
        cust = gt.get("customer")
        if cust not in specs:
            print(f"! {gt_path.name}: customer {cust!r} not in the spec — skipped", file=sys.stderr)
            continue
        s = score_file(specs[cust], pred, gt)
        s.update(file=gt_path.stem, customer=cust, pred_customer=pred.get("customer"), error=pred.get("error", ""))
        files.append(s)
        for r in s["fields"]:
            for key, agg in ((f"{cust} / {r['field']}", per_field), (cust, per_customer)):
                agg[key][0] += r["strict"]
                agg[key][1] += r["loose"]
                agg[key][2] += 1
    n_fields = sum(len(f["fields"]) for f in files)
    return {
        "n_files": len(files),
        "customer_acc": _ratio(sum(f["customer_ok"] for f in files), len(files)),
        "field_acc_strict": _ratio(sum(r["strict"] for f in files for r in f["fields"]), n_fields),
        "field_acc_loose": _ratio(sum(r["loose"] for f in files for r in f["fields"]), n_fields),
        "file_exact": _ratio(sum(f["exact"] for f in files), len(files)),
        "spec_conformity": _ratio(sum(f["conform"] for f in files), len(files)),
        "per_customer": {k: {"strict": _ratio(v[0], v[2]), "loose": _ratio(v[1], v[2]), "n": v[2]}
                         for k, v in sorted(per_customer.items())},
        "per_field": {k: {"strict": _ratio(v[0], v[2]), "loose": _ratio(v[1], v[2]), "n": v[2]}
                      for k, v in sorted(per_field.items())},
        "files": files,
    }


def _ratio(a, b):
    return round(a / b, 4) if b else None


def report_md(rep: dict) -> str:
    def pct(x):
        return "-" if x is None else f"{x:.1%}"

    out = ["# PO extraction — evaluation", "",
           f"- files: {rep['n_files']}",
           f"- customer detection: {pct(rep['customer_acc'])}",
           f"- field accuracy (strict / loose): {pct(rep['field_acc_strict'])} / {pct(rep['field_acc_loose'])}",
           f"- files with every field correct: {pct(rep['file_exact'])}",
           f"- spec conformity (exactly the customer's fields): {pct(rep['spec_conformity'])}", "",
           "## Per customer", "", "| customer | n fields | strict | loose |", "|---|---|---|---|"]
    out += [f"| {k} | {v['n']} | {pct(v['strict'])} | {pct(v['loose'])} |" for k, v in rep["per_customer"].items()]
    out += ["", "## Errors", "", "| file | field | predicted | expected |", "|---|---|---|---|"]
    for f in rep["files"]:
        if f["error"]:
            out.append(f"| {f['file']} | (error) | {f['error']} | |")
        if not f["customer_ok"]:
            out.append(f"| {f['file']} | (customer) | {f['pred_customer']} | {f['customer']} |")
        for r in f["fields"]:
            if not r["strict"]:
                p = json.dumps(r["pred"], ensure_ascii=False)[:120].replace("|", "/")
                g = json.dumps(r["gt"], ensure_ascii=False)[:120].replace("|", "/")
                out.append(f"| {f['file']} | {r['field']} | {p} | {g} |")
    return "\n".join(out) + "\n"


def init_gt(pred_dir: Path, gt_dir: Path, overwrite: bool = False) -> int:
    """Copy predictions into GT templates (`checked: false`) for a human to correct."""
    gt_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(pred_dir.glob("*.json")):
        target = gt_dir / p.name
        if target.exists() and not overwrite:
            continue
        pred = json.loads(p.read_text(encoding="utf-8"))
        gt = {"file": pred.get("file"), "customer": pred.get("customer"), "checked": False,
              "fields": pred.get("fields", {})}
        target.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
        n += 1
    return n


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("init-gt", help="create GT templates from predictions")
    a.add_argument("--pred", default=str(ROOT / "results" / "po"))
    a.add_argument("--gt", default=str(ROOT / "data" / "po_gt"))
    a.add_argument("--overwrite", action="store_true")
    b = sub.add_parser("score", help="score predictions against GT")
    b.add_argument("--pred", default=str(ROOT / "results" / "po"))
    b.add_argument("--gt", default=str(ROOT / "data" / "po_gt"))
    b.add_argument("--spec", help="YAML/Excel spec (default: config/po_customers.yaml)")
    b.add_argument("--only-checked", action="store_true", help="ignore GT files still marked checked: false")
    b.add_argument("--out", default=str(ROOT / "results" / "po_eval"))
    args = ap.parse_args(argv)

    if args.cmd == "init-gt":
        print(f"{init_gt(Path(args.pred), Path(args.gt), args.overwrite)} GT templates written to {args.gt}")
        return
    spec_path = args.spec or default_spec_path()
    if not spec_path:
        sys.exit("No spec file: pass --spec")
    rep = score_dir(Path(args.pred), Path(args.gt), load_specs(spec_path), args.only_checked)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    md = report_md(rep)
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
