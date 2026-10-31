"""Stage A: retrieval-only benchmark over (chunking x retriever [x reranker]) configs.

    uv run python -m bibica_rag.eval.stage_a --phase A1 --workers 4
Results: results/stageA/<config>.json (per-query) + results/stageA.csv (aggregate).
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ..corpus import ROOT
from ..testset import load_testset

OUT = ROOT / "results" / "stageA"
CSV = ROOT / "results" / "stageA.csv"

CHUNKINGS = ["fixed500", "fixed1000", "recursive1000", "sentence600", "semantic", "article", "clause_noheader",
             "clause_header", "clause_parent"]
DENSE = ["bge-m3", "vn-embedding", "qwen3-emb-0.6b", "me5-large-instruct", "halong"]


def build_chunks(name):
    from ..chunking import STRATEGIES
    from ..retrieval import DenseEmbedder

    if name == "semantic":
        return STRATEGIES[name](embed_fn=DenseEmbedder("bge-m3").embed_docs)
    if name.startswith("ctx_"):  # LLM contextual retrieval chunks, produced by bibica_rag.contextualize
        from ..chunking import contextual_llm

        base = STRATEGIES[name[len("ctx_"):]]()
        ctx = json.loads((ROOT / "data" / "cache" / f"contexts_{name[len('ctx_'):]}.json").read_text())
        return contextual_llm(base, ctx)
    return STRATEGIES[name]()


def config_id(chunking, spec):
    return f"{chunking}__{spec}".replace("/", "_").replace("|", "__").replace(":", "-").replace("+", "_")


def run_config(chunking, spec, threads=4):
    import torch

    torch.set_num_threads(threads)
    from ..retrieval import build_retriever
    from .retrieval_metrics import aggregate, metrics_for, ranked_nodes

    qs = [q for q in load_testset() if q["answerable"]]
    t0 = time.time()
    chunks = build_chunks(chunking)
    retr = build_retriever(spec, chunks)
    t_index = time.time() - t0
    t1 = time.time()
    rankings = retr.search([q["question"] for q in qs], k=20)
    t_query = (time.time() - t1) / len(qs)
    rows, per_q = [], []
    for q, rk in zip(qs, rankings):
        nodes = ranked_nodes(rk, chunks)
        m = metrics_for(nodes, q["gold_nodes"])
        m["ctx_chars@5"] = sum(len(chunks[i]["context"]) for i, _ in rk[:5])
        rows.append(m)
        per_q.append({"id": q["id"], "type": q["type"], "top_nodes": nodes[:10], **m})
    agg = aggregate(rows)
    res = {"config": config_id(chunking, spec), "chunking": chunking, "retriever": spec, "n_chunks": len(chunks),
           "n_questions": len(qs), "index_s": round(t_index, 2), "query_ms": round(1000 * t_query, 1), **agg}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{res['config']}.json").write_text(json.dumps({"summary": res, "per_question": per_q},
                                                          ensure_ascii=False, indent=1))
    return res


def phase_configs(phase, top_chunkings=None, top_retrievers=None):
    if phase == "A1":  # chunking sweep with 3 representative retrievers
        return [(c, s) for c in CHUNKINGS for s in ["dense:bge-m3", "dense:vn-embedding", "bm25:foldplus"]]
    if phase == "A2":  # retriever sweep on best chunkings
        specs = [f"dense:{d}" for d in DENSE] + ["bm25:syl", "bm25:fold", "bm25:pyvi", "bm25:foldplus",
                                                  "bgem3all",
                                                  "hybrid:dense:vn-embedding+bm25:foldplus",
                                                  "hybrid:dense:bge-m3+bm25:foldplus",
                                                  "hybrid:dense:qwen3-emb-0.6b+bm25:foldplus",
                                                  "hybrid:dense:vn-embedding+dense:qwen3-emb-0.6b+bm25:foldplus"]
        return [(c, s) for c in top_chunkings for s in specs]
    if phase == "A3":  # rerankers on best (chunking, retriever) pairs
        return [(c, f"{s}|rerank:{r}") for c, s in top_retrievers
                for r in ["bge-reranker-v2-m3", "vn-reranker", "qwen3-reranker-0.6b"]]
    raise ValueError(phase)


def leaderboard():
    rows = [json.loads(p.read_text())["summary"] for p in OUT.glob("*.json")]
    df = pd.DataFrame(rows).sort_values(["recall@5", "mrr@10"], ascending=False)
    df.to_csv(CSV, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--chunkings", nargs="*", help="override the automatic top-N chunking choice for A2")
    ap.add_argument("--bases", nargs="*", help="A3 base configs as chunking::retriever (overrides --top)")
    args = ap.parse_args()
    df = leaderboard() if any(OUT.glob("*.json")) else None
    top_chunk = top_ret = None
    if args.phase == "A2":
        a1 = df[df.chunking.isin(CHUNKINGS)].groupby("chunking")["recall@5"].mean().sort_values(ascending=False)
        top_chunk = args.chunkings or list(a1.index[: args.top])
        print("top chunkings:", top_chunk)
    if args.phase == "A3":
        base = df[~df.retriever.str.contains("rerank")]
        top_ret = ([tuple(b.split("::")) for b in args.bases] if args.bases else
                   list(base.head(args.top)[["chunking", "retriever"]].itertuples(index=False, name=None)))
        print("top retrievers:", top_ret)
    configs = phase_configs(args.phase, top_chunk, top_ret)
    todo = [c for c in configs if args.force or not (OUT / f"{config_id(*c)}.json").exists()]
    print(f"{len(configs)} configs, {len(todo)} to run")
    threads = max(1, (os.cpu_count() or 8) // args.workers)
    # group by retriever spec so each worker reuses loaded models
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_config, c, s, threads): (c, s) for c, s in sorted(todo, key=lambda x: x[1])}
        for f in as_completed(futs):
            try:
                r = f.result()
                print(f"{r['config']:70s} R@5={r['recall@5']:.3f} MRR={r['mrr@10']:.3f} q={r['query_ms']}ms",
                      flush=True)
            except Exception as e:  # keep the sweep going
                print("FAILED", futs[f], repr(e), flush=True)
    print(leaderboard().head(15).to_string())


if __name__ == "__main__":
    main()
