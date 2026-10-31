"""Aggregate end-to-end runs + judge verdicts into a leaderboard.

    uv run python -m bibica_rag.eval.report            -> results/leaderboard.csv, results/leaderboard.json

Pass rule (per test case):
  answerable   : judge correctness == 1 AND judge faithful AND every key fact present in the answer
  unanswerable : judge correctness == 1 (correct refusal) AND faithful
Latency comes from the sequential latency runs (results/latency/<run>/, 1 worker, no contention) when present;
otherwise from the benchmark run itself (4 concurrent requests, so slower than a single user would see).
"""
import json
import statistics

import numpy as np

from ..corpus import ROOT
from .judge import akey, load_verdicts

RUNS = ROOT / "results" / "runs"
LAT = ROOT / "results" / "latency"
OUT = ROOT / "results"


def _records(d):
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json")) if not p.name.startswith("_")]


def _pct(xs, p):
    return float(np.percentile(xs, p)) if xs else None


def passed(r, v):
    if v is None:
        return None
    if r["answerable"]:
        return v["correctness"] == 1 and v["faithful"] and bool(r.get("keyfact_all"))
    return v["correctness"] == 1 and v["faithful"]


def summarize(run_dir, verdicts, split_ids=None):
    meta = json.loads((run_dir / "_meta.json").read_text(encoding="utf-8"))
    recs = [r for r in _records(run_dir) if split_ids is None or r["id"] in split_ids]
    if not recs:
        return None
    vs = [verdicts.get(akey(r["id"], r["answer"])) for r in recs]
    judged = [(r, v) for r, v in zip(recs, vs) if v is not None]
    ps = [passed(r, v) for r, v in judged]
    ans = [(r, v) for r, v in judged if r["answerable"]]
    una = [(r, v) for r, v in judged if not r["answerable"]]
    ret = [r["retrieval"] for r in recs if r.get("retrieval")]
    lat_dir = LAT / run_dir.name
    lat_recs = _records(lat_dir) if lat_dir.exists() else []
    lat_src = "sequential" if lat_recs else "concurrent(4)"
    tt = [r["timings"]["total_s"] for r in (lat_recs or recs)]
    ttft = [r["timings"]["ttft_s"] for r in (lat_recs or recs) if r["timings"].get("ttft_s") is not None]
    row = {
        "run": run_dir.name, "name": meta.get("name"), "framework": meta.get("framework"),
        "technique": meta.get("technique"), "chunking": meta.get("chunking"), "retriever": meta.get("retriever"),
        "model": meta.get("model"), "n": len(recs), "n_judged": len(judged),
        "pass": sum(bool(p) for p in ps), "pass_rate": sum(bool(p) for p in ps) / len(judged) if judged else None,
        "correctness": statistics.mean(v["correctness"] for _, v in judged) if judged else None,
        "faithfulness": statistics.mean(bool(v["faithful"]) for _, v in judged) if judged else None,
        "citation_ok": (statistics.mean(bool(v["citation_ok"]) for _, v in judged if v.get("citation_ok") is not None)
                        if any(v.get("citation_ok") is not None for _, v in judged) else None),
        "keyfact_all": statistics.mean(bool(r.get("keyfact_all")) for r in recs if r["answerable"]),
        "refusal_acc": statistics.mean(v["correctness"] == 1 for _, v in una) if una else None,
        "answerable_correct": statistics.mean(v["correctness"] for _, v in ans) if ans else None,
        "recall@5": statistics.mean(m["recall@5"] for m in ret) if ret else None,
        "mrr@10": statistics.mean(m["mrr@10"] for m in ret) if ret else None,
        "latency_src": lat_src, "latency_p50_s": _pct(tt, 50), "latency_p95_s": _pct(tt, 95),
        "ttft_p50_s": _pct(ttft, 50), "llm_calls": statistics.mean(r.get("llm_calls", 1) for r in recs),
        "build_s": meta.get("build_s"),
    }
    by = {}
    for (r, v), p in zip(judged, ps):
        for key in (f"type:{r['type']}", f"doc:{r['doc']}"):
            by.setdefault(key, []).append(bool(p))
    row["breakdown"] = {k: [sum(x), len(x)] for k, x in sorted(by.items())}
    return row


def main(split="test"):
    from ..testset import load_testset
    ids = {q["id"] for q in load_testset(split)}
    verdicts = load_verdicts()
    rows = [s for d in sorted(RUNS.iterdir()) if d.is_dir() and (d / "_meta.json").exists()
            for s in [summarize(d, verdicts, ids)] if s]
    # results/runs/ is not tracked: keep leaderboard rows of runs whose answers are not on this machine
    # (e.g. the Qwen3.5-4B Stage-B runs when only the 9B Stage-C runs were copied here).
    old = OUT / "leaderboard.json"
    if old.exists():
        have = {r["run"] for r in rows}
        rows += [r for r in json.loads(old.read_text(encoding="utf-8")) if r["run"] not in have]
    rows.sort(key=lambda r: (-(r["pass"] or 0), -(r["correctness"] or 0), r["latency_p50_s"] or 1e9))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    (OUT / "leaderboard.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    import pandas as pd
    pd.DataFrame([{k: v for k, v in r.items() if k != "breakdown"} for r in rows]).to_csv(OUT / "leaderboard.csv",
                                                                                          index=False)
    for r in rows:
        print(f"{r['rank']:2d}. {r['run'][:60]:60s} pass {r['pass']}/{r['n_judged']}  corr {r['correctness'] or 0:.3f} "
              f"faith {r['faithfulness'] or 0:.3f}  p50 {r['latency_p50_s'] or 0:.1f}s")


if __name__ == "__main__":
    main()
