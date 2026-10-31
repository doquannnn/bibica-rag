"""Node-level retrieval metrics: chunks are mapped to the Điều/Phụ lục nodes they cover."""
import math


def ranked_nodes(ranking, chunks, k=None):
    """Chunk ranking -> de-duplicated node ranking (a chunk spanning several nodes yields them in order)."""
    seen, out = set(), []
    for idx, _ in ranking:
        for nid in chunks[idx]["node_ids"]:
            if nid not in seen:
                seen.add(nid)
                out.append(nid)
    return out[:k] if k else out


def metrics_for(nodes, gold):
    gold = set(gold)
    m = {}
    for k in (1, 3, 5, 10):
        top = nodes[:k]
        m[f"hit@{k}"] = float(any(n in gold for n in top))
        m[f"recall@{k}"] = len(gold & set(top)) / len(gold)
    rr = 0.0
    for i, n in enumerate(nodes[:10]):
        if n in gold:
            rr = 1.0 / (i + 1)
            break
    m["mrr@10"] = rr
    dcg = sum(1.0 / math.log2(i + 2) for i, n in enumerate(nodes[:10]) if n in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    m["ndcg@10"] = dcg / idcg
    return m


def aggregate(rows):
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}
