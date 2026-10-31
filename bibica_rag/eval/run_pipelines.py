"""Run end-to-end pipelines over a test split with per-question caching (resumable).

    uv run python -m bibica_rag.eval.run_pipelines --config configs/stageB.yaml --split test --workers 4
Outputs results/runs/<run_name>/<qid>.json and results/runs/<run_name>/_meta.json
"""
import argparse
import json
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from ..corpus import ROOT
from ..llm import BASE_URL, MODEL, MODEL_LABEL
from ..testset import load_testset
from .keyfacts import score
from .retrieval_metrics import metrics_for

RUNS = ROOT / "results" / "runs"


def build_pipeline(spec: dict):
    kind = spec["kind"]
    args = {k: v for k, v in spec.items() if k not in ("kind", "run")}
    if kind == "naive":
        from ..pipelines.base import NaiveRAG
        return NaiveRAG(**args)
    if kind in ("multiquery", "hyde"):
        from ..pipelines.base import QueryTransformRAG
        return QueryTransformRAG(mode=kind, **args)
    if kind == "cag":
        from ..pipelines.base import FullContextCAG
        return FullContextCAG()
    if kind == "langchain":
        from ..pipelines.frameworks import LangChainRAG
        return LangChainRAG(**args)
    if kind == "langgraph_crag":
        from ..pipelines.frameworks import LangGraphCRAG
        return LangGraphCRAG(**args)
    if kind == "llamaindex":
        from ..pipelines.frameworks import LlamaIndexRAG
        return LlamaIndexRAG(**args)
    if kind == "haystack":
        from ..pipelines.frameworks import HaystackRAG
        return HaystackRAG(**args)
    if kind == "dspy":
        from ..pipelines.frameworks import DSPyRAG
        return DSPyRAG(**args)
    if kind == "raptor":
        from ..pipelines.raptor import RaptorRAG
        return RaptorRAG(**args)
    if kind == "llm_wiki":
        from ..pipelines.llm_wiki import LLMWiki
        return LLMWiki(**args)
    if kind == "lightrag":
        from ..pipelines.lightrag_pipe import LightRAGPipe
        return LightRAGPipe(**args)
    raise ValueError(kind)


def run_name(spec):
    return spec.get("run") or re.sub(r"[^\w.+-]+", "_", json.dumps(spec, sort_keys=True))[:120]


def run_one(spec, split, workers, limit=None, root=RUNS):
    name = f"{run_name(spec)}@{MODEL_LABEL}"
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    qs = load_testset(split)[:limit] if limit else load_testset(split)
    todo = [q for q in qs if not (out_dir / f"{q['id']}.json").exists()]
    print(f"== {name}: {len(qs)} questions, {len(todo)} to run", flush=True)
    if not todo:
        return name
    t0 = time.time()
    pipe = build_pipeline(spec)
    t_build = time.time() - t0
    meta = {"run": name, "spec": spec, "model": MODEL_LABEL, "model_id": MODEL, "endpoint": BASE_URL, "split": split, **pipe.describe(), "build_s": round(t_build, 1)}
    (out_dir / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    def work(q):
        try:
            r = pipe.answer(q["question"])
        except Exception as e:
            return q, {"error": repr(e), "trace": traceback.format_exc()[-2000:]}
        r.update(score(q, r["answer"]))
        if q["answerable"] and r.get("retrieved_nodes"):
            r["retrieval"] = metrics_for(r["retrieved_nodes"], q["gold_nodes"])
        r["contexts"] = [{"node_ids": c["node_ids"], "context": c["context"][:6000]} for c in r["contexts"]]
        return q, r

    done = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(work, q) for q in todo]
        for f in as_completed(futs):
            q, r = f.result()
            if "error" in r:
                print("ERROR", q["id"], r["error"], flush=True)
                continue
            rec = {"id": q["id"], "question": q["question"], "type": q["type"], "doc": q["doc"],
                   "answerable": q["answerable"], **r}
            (out_dir / f"{q['id']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
            done += 1
            if done % 20 == 0:
                print(f"   {done}/{len(todo)}  {(time.time() - t0) / done:.1f}s/q wall", flush=True)
    print(f"   finished {name} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", nargs="*", help="run names to run")
    ap.add_argument("--latency", action="store_true",
                    help="sequential latency run: 1 worker, first 30 test questions, saved under results/latency/")
    args = ap.parse_args()
    root = RUNS
    if args.latency:
        root, args.workers, args.limit = ROOT / "results" / "latency", 1, args.limit or 30
    specs = yaml.safe_load(open(args.config))
    for spec in specs:
        if args.only and spec.get("run") not in args.only:
            continue
        run_one(spec, args.split, args.workers, args.limit, root)


if __name__ == "__main__":
    main()
