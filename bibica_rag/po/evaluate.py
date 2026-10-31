"""Accuracy of the PO extractor against hand-checked ground truth.

Ground truth: one JSON per PDF in `data/po_gt/<pdf stem>.json`:
    {"customer": "<name as in the spec>", "checked": true, "fields": {"<field>": <value or null>,
     "<table field>": [{...}, ...]}}

Workflow
    python -m bibica_rag.po.extract "PO Extraction/" --out results/po
    python -m bibica_rag.po.evaluate init-gt --pred results/po --gt data/po_gt   # new PDFs only; correct by hand
    python -m bibica_rag.po.evaluate score --pred results/po --gt data/po_gt     # results/po_eval/report.md

Metrics (what matters for creating the order in the DMS):
- customer detection accuracy
- header fields (Số PO, ngày giao, mã KH): accuracy, strict (after type normalization) and loose
  (accents/punctuation ignored)
- line items, matched by product code (the first text column, e.g. "Mã SP") — not by position, so one
  skipped line does not make every following line wrong:
    line precision = correct codes / predicted lines, recall = correct codes / GT lines, F1;
    per-column accuracy on matched lines (e.g. "SL đặt"); a line is exact when the code and every column match
- order exact match: every header field and every line exactly right — the share of POs that would go into
  the DMS with no correction
- spec conformity: the output keys are exactly the customer's fields
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


def _key_col(f: Field) -> Field | None:
    return next((c for c in f.columns if c.type == "text"), f.columns[0] if f.columns else None)


def compare_table(f: Field, pred, gt) -> dict:
    """Match lines by key (product code), greedily in document order; duplicates are matched one to one."""
    pred, gt = pred or [], gt or []
    key = _key_col(f)
    cols = [c for c in f.columns if c is not key]
    kn = (lambda r: slug(normalize_value(r.get(key.name), "text") or "")) if key else (lambda r: "")
    free = defaultdict(list)
    for j, g in enumerate(gt):
        free[kn(g)].append(j)
    pairs = []
    for i, p in enumerate(pred):
        k = kn(p)
        if k and free.get(k):
            pairs.append((i, free[k].pop(0)))
    col_hits = {c.name: sum(values_equal(pred[i].get(c.name), gt[j].get(c.name), c.type) for i, j in pairs)
                for c in cols}
    exact = sum(all(values_equal(pred[i].get(c.name), gt[j].get(c.name), c.type) for c in cols) for i, j in pairs)
    matched_gt = {j for _, j in pairs}
    return {"n_pred": len(pred), "n_gt": len(gt), "matched": len(pairs), "exact": exact, "col_hits": col_hits,
            "missed": [gt[j].get(key.name) for j in range(len(gt)) if j not in matched_gt] if key else [],
            "extra": [pred[i].get(key.name) for i in range(len(pred)) if i not in {a for a, _ in pairs}] if key
            else [],
            "wrong": [{"pred": pred[i], "gt": gt[j]} for i, j in pairs
                      if not all(values_equal(pred[i].get(c.name), gt[j].get(c.name), c.type) for c in cols)]}


def score_file(spec: CustomerSpec, pred: dict, gt: dict) -> dict:
    pf, gf = pred.get("fields", {}), gt.get("fields", {})
    header, tables = [], {}
    for f in spec.fields:
        if f.type == "table":
            tables[f.name] = compare_table(f, pf.get(f.name), gf.get(f.name))
        else:
            header.append({"field": f.name, "type": f.type, "strict": values_equal(pf.get(f.name), gf.get(f.name), f.type),
                           "loose": values_equal(pf.get(f.name), gf.get(f.name), f.type, loose=True),
                           "pred": pf.get(f.name), "gt": gf.get(f.name)})
    lines_exact = all(t["exact"] == t["n_gt"] == t["n_pred"] for t in tables.values())
    return {"customer_ok": pred.get("customer") == gt.get("customer"),
            "conform": list(pf) == spec.field_names, "header": header, "tables": tables,
            "exact": bool(pf) and all(r["strict"] for r in header) and lines_exact}


def _f1(p, r):
    return round(2 * p * r / (p + r), 4) if p and r else 0.0


def score_dir(pred_dir: Path, gt_dir: Path, specs: dict[str, CustomerSpec], only_checked: bool = False) -> dict:
    files = []
    per_field = defaultdict(lambda: [0, 0, 0])  # strict, loose, n
    per_customer = defaultdict(lambda: {"files": 0, "exact": 0, "header_ok": 0, "header_n": 0, "gt_lines": 0,
                                        "pred_lines": 0, "matched": 0, "exact_lines": 0})
    col_hits, col_n = defaultdict(int), defaultdict(int)
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
        s.update(file=gt_path.stem, customer=cust, pred_customer=pred.get("customer"),
                 error=pred.get("error", "") if pred else "no prediction", warnings=pred.get("warnings", []))
        files.append(s)
        pc = per_customer[cust]
        pc["files"] += 1
        pc["exact"] += s["exact"]
        for r in s["header"]:
            agg = per_field[r["field"]]
            agg[0] += r["strict"]
            agg[1] += r["loose"]
            agg[2] += 1
            pc["header_ok"] += r["strict"]
            pc["header_n"] += 1
        for t in s["tables"].values():
            pc["gt_lines"] += t["n_gt"]
            pc["pred_lines"] += t["n_pred"]
            pc["matched"] += t["matched"]
            pc["exact_lines"] += t["exact"]
            for c, h in t["col_hits"].items():
                col_hits[c] += h
                col_n[c] += t["matched"]
    tot = {k: sum(v[k] for v in per_customer.values()) for k in
           ("files", "exact", "header_ok", "header_n", "gt_lines", "pred_lines", "matched", "exact_lines")}
    prec, rec = _ratio(tot["matched"], tot["pred_lines"]), _ratio(tot["matched"], tot["gt_lines"])
    n_header = sum(len(f["header"]) for f in files)
    return {
        "n_files": len(files),
        "customer_acc": _ratio(sum(f["customer_ok"] for f in files), len(files)),
        "header_acc_strict": _ratio(sum(r["strict"] for f in files for r in f["header"]), n_header),
        "header_acc_loose": _ratio(sum(r["loose"] for f in files for r in f["header"]), n_header),
        "line_precision": prec, "line_recall": rec, "line_f1": _f1(prec or 0, rec or 0),
        "line_exact_rate": _ratio(tot["exact_lines"], tot["gt_lines"]),
        "column_acc_on_matched": {c: _ratio(col_hits[c], col_n[c]) for c in col_hits},
        "order_exact": _ratio(sum(f["exact"] for f in files), len(files)),
        "spec_conformity": _ratio(sum(f["conform"] for f in files), len(files)),
        "per_customer": {k: {"files": v["files"], "order_exact": _ratio(v["exact"], v["files"]),
                             "header_acc": _ratio(v["header_ok"], v["header_n"]),
                             "line_recall": _ratio(v["matched"], v["gt_lines"]),
                             "line_precision": _ratio(v["matched"], v["pred_lines"]),
                             "line_exact": _ratio(v["exact_lines"], v["gt_lines"]), "gt_lines": v["gt_lines"]}
                         for k, v in sorted(per_customer.items())},
        "per_field": {k: {"strict": _ratio(v[0], v[2]), "loose": _ratio(v[1], v[2]), "n": v[2]}
                      for k, v in sorted(per_field.items())},
        "files": files,
    }


def _ratio(a, b):
    return round(a / b, 4) if b else None


def report_md(rep: dict, title: str = "PO extraction — evaluation") -> str:
    def pct(x):
        return "-" if x is None else f"{x:.1%}"

    cols = ", ".join(f"{k} {pct(v)}" for k, v in rep["column_acc_on_matched"].items()) or "-"
    out = [f"# {title}", "",
           f"- files: {rep['n_files']} · customer detection: {pct(rep['customer_acc'])} · "
           f"spec conformity: {pct(rep['spec_conformity'])}",
           f"- **orders fully correct (no correction needed in DMS): {pct(rep['order_exact'])}**",
           f"- header fields (strict / loose): {pct(rep['header_acc_strict'])} / {pct(rep['header_acc_loose'])}",
           f"- line items by product code: precision {pct(rep['line_precision'])} · recall "
           f"{pct(rep['line_recall'])} · F1 {pct(rep['line_f1'])} · lines fully correct {pct(rep['line_exact_rate'])}",
           f"- columns on matched lines: {cols}", "",
           "## Per customer", "",
           "| customer | files | order exact | header | line recall | line precision | lines exact | GT lines |",
           "|---|---|---|---|---|---|---|---|"]
    out += [f"| {k} | {v['files']} | {pct(v['order_exact'])} | {pct(v['header_acc'])} | {pct(v['line_recall'])} | "
            f"{pct(v['line_precision'])} | {pct(v['line_exact'])} | {v['gt_lines']} |"
            for k, v in rep["per_customer"].items()]
    out += ["", "## Per header field", "", "| field | n | strict | loose |", "|---|---|---|---|"]
    out += [f"| {k} | {v['n']} | {pct(v['strict'])} | {pct(v['loose'])} |" for k, v in rep["per_field"].items()]
    out += ["", "## Errors", "", "| file | where | predicted | expected |", "|---|---|---|---|"]

    def j(x):
        return json.dumps(x, ensure_ascii=False)[:160].replace("|", "/")
    for f in rep["files"]:
        if f["error"]:
            out.append(f"| {f['file']} | (error) | {f['error'][:160]} | |")
        if not f["customer_ok"]:
            out.append(f"| {f['file']} | (customer) | {f['pred_customer']} | {f['customer']} |")
        for r in f["header"]:
            if not r["strict"]:
                out.append(f"| {f['file']} | {r['field']} | {j(r['pred'])} | {j(r['gt'])} |")
        for name, t in f["tables"].items():
            if t["missed"]:
                out.append(f"| {f['file']} | {name}: missed lines | | {j(t['missed'])} |")
            if t["extra"]:
                out.append(f"| {f['file']} | {name}: extra/misread codes | {j(t['extra'])} | |")
            for w in t["wrong"]:
                out.append(f"| {f['file']} | {name}: wrong value | {j(w['pred'])} | {j(w['gt'])} |")
    warned = [f for f in rep["files"] if f["warnings"]]
    if warned:
        out += ["", "## Self-check warnings (validate.py)", ""]
        out += [f"- {f['file']}: " + "; ".join(f["warnings"]) for f in warned]
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
