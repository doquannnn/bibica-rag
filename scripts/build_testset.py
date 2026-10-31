"""Merge the verified question files and split them into dev (60) / test (rest), stratified by (type, doc)."""
import collections
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data" / "testset"
FIELDS = ["id", "question", "type", "doc", "answerable", "reference_answer", "gold_nodes", "key_facts"]
N_DEV = 60

rows = []
for name in ["verified_QCL", "verified_NQLD", "verified_TULD_MULTI"]:
    rows += [json.loads(l) for l in open(D / f"{name}.jsonl", encoding="utf-8") if l.strip()]
rows = [{k: r[k] for k in FIELDS} for r in rows]
assert len({r["id"] for r in rows}) == len(rows)

# deterministic stratified sample: within each (type, doc) stratum order by hash, take a proportional share
strata = collections.defaultdict(list)
for r in rows:
    strata[(r["type"], r["doc"])].append(r)
dev = []
for key, items in sorted(strata.items()):
    items.sort(key=lambda r: hashlib.md5(r["id"].encode()).hexdigest())
    take = round(len(items) * N_DEV / len(rows))
    dev += items[:take]
dev_ids = {r["id"] for r in dev}
test = [r for r in rows if r["id"] not in dev_ids]

for name, part in [("dev", dev), ("test", test)]:
    with open(D / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(part), dict(collections.Counter(r["type"] for r in part)))
print("total", len(rows), dict(collections.Counter(r["doc"] for r in rows)))
