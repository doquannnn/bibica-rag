"""LLM-as-judge with Claude subagents: export answer batches grouped by question, import verdicts, cache by hash.

    uv run python -m bibica_rag.eval.judge export --runs <run dirs...> --per-batch 10
    (Claude subagents judge results/judge/batches/*.json -> results/judge/out/*.jsonl following RUBRIC.md)
    uv run python -m bibica_rag.eval.judge import

A verdict is cached under sha1(question id + answer), so identical answers from different runs are judged once and
re-running a pipeline only needs judging of answers that changed.
"""
import argparse
import hashlib
import json

from ..corpus import ROOT, node_by_id
from ..testset import load_testset

J = ROOT / "results" / "judge"
RUNS = ROOT / "results" / "runs"
CACHE_FILE = J / "verdicts.jsonl"


def akey(qid, answer):
    return hashlib.sha1(f"{qid}\x00{answer.strip()}".encode()).hexdigest()[:16]


def load_verdicts():
    out = {}
    if CACHE_FILE.exists():
        for line in CACHE_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                v = json.loads(line)
                out[v["key"]] = v
    return out


def export(run_names, per_batch=10):
    qs = {q["id"]: q for q in load_testset()}
    nodes = node_by_id()
    done = load_verdicts()
    pending = {}  # qid -> {key: answer}
    for rn in run_names:
        for p in sorted((RUNS / rn).glob("*.json")):
            if p.name.startswith("_"):
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            k = akey(r["id"], r["answer"])
            if k not in done:
                pending.setdefault(r["id"], {})[k] = r["answer"]
    (J / "batches").mkdir(parents=True, exist_ok=True)
    (J / "out").mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in (J / "batches").glob("*.json")}
    start = max([int(s.split("_")[1]) for s in existing] + [0]) + 1
    qids = sorted(pending)
    made = []
    for i in range(0, len(qids), per_batch):
        items = []
        for qid in qids[i:i + per_batch]:
            q = qs[qid]
            items.append({
                "qid": qid, "question": q["question"], "type": q["type"], "answerable": q["answerable"],
                "reference_answer": q["reference_answer"], "key_facts": q["key_facts"],
                "gold_evidence": {n: nodes[n]["title"] + "\n" + nodes[n]["text"] for n in q["gold_nodes"]},
                "answers": [{"key": k, "answer": a} for k, a in pending[qid].items()],
            })
        name = f"batch_{start + len(made):04d}"
        (J / "batches" / f"{name}.json").write_text(json.dumps(items, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
        made.append(name)
    n_ans = sum(len(v) for v in pending.values())
    print(f"{n_ans} answers to judge over {len(qids)} questions -> {len(made)} batches: {made[:3]}...{made[-1:]}")
    return made


def import_verdicts():
    done = load_verdicts()
    new = 0
    with open(CACHE_FILE, "a", encoding="utf-8") as f:
        # rejudge_*.jsonl (re-grading after a rubric fix) sort after batch_* and override earlier verdicts
        for p in sorted((J / "out").glob("*.jsonl")):
            override = p.name.startswith("rejudge_")
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                v = json.loads(line)
                if override and done.get(v["key"]) != v:
                    done[v["key"]] = v
                    f.write(json.dumps(v, ensure_ascii=False) + "\n")
                    new += 1
                elif v["key"] not in done:
                    done[v["key"]] = v
                    f.write(json.dumps(v, ensure_ascii=False) + "\n")
                    new += 1
    print(f"imported {new} new verdicts, {len(done)} total")


def export_rejudge(qids, name):
    """All answers (across every run) to the given questions, regardless of existing verdicts."""
    qs = {q["id"]: q for q in load_testset()}
    nodes = node_by_id()
    items = []
    for qid in qids:
        answers = {}
        for p in RUNS.glob(f"*/{qid}.json"):
            r = json.loads(p.read_text(encoding="utf-8"))
            answers[akey(qid, r["answer"])] = r["answer"]
        q = qs[qid]
        items.append({"qid": qid, "question": q["question"], "type": q["type"], "answerable": q["answerable"],
                      "reference_answer": q["reference_answer"], "key_facts": q["key_facts"],
                      "gold_evidence": {n: nodes[n]["title"] + "\n" + nodes[n]["text"] for n in q["gold_nodes"]},
                      "answers": [{"key": k, "answer": a} for k, a in answers.items()]})
    (J / "batches" / f"{name}.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{name}: {sum(len(i['answers']) for i in items)} answers")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "import", "rejudge"])
    ap.add_argument("--qids", nargs="*", default=[])
    ap.add_argument("--name", default="rejudge_0001")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--per-batch", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.runs or sorted(p.name for p in RUNS.iterdir() if p.is_dir()), a.per_batch)
    elif a.cmd == "rejudge":
        export_rejudge(a.qids, a.name)
    else:
        import_verdicts()
