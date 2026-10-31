"""Load the benchmark test set (data/testset/{dev,test}.jsonl)."""
import json

from .corpus import DATA

TESTSET = DATA / "testset"


def load_testset(split: str = "all"):
    splits = ["dev", "test"] if split == "all" else [split]
    out = []
    for s in splits:
        p = TESTSET / f"{s}.jsonl"
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                q = json.loads(line)
                q["split"] = s
                out.append(q)
    return out
