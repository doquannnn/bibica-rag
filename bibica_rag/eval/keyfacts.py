"""Deterministic key-fact matching: numbers / percentages / named terms must appear in the answer."""
import re

from ..corpus import strip_diacritics

REFUSAL = re.compile(r"không có thông tin|không có quy định|không đề cập|không được đề cập|không nêu|không tìm thấy|"
                     r"không chứa thông tin|không có dữ liệu|chưa có quy định|không quy định", re.I)


def norm(s: str) -> str:
    s = strip_diacritics(s.lower())
    s = s.replace(",", ".")
    s = re.sub(r"(?<!\d)0+(\d)", r"\1", s)  # 06 -> 6, 05 chương -> 5 chương
    s = re.sub(r"\s*%", "%", s)
    s = re.sub(r"[^\w%./<>=+-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def fact_present(fact: str, answer: str) -> bool:
    f, a = norm(fact), norm(answer)
    if not f:
        return True
    if f in a:
        return True
    # numbers: every number in the fact must appear as a standalone number in the answer
    nums = re.findall(r"\d+(?:\.\d+)?%?", f)
    words = [w for w in re.findall(r"[a-z]+", f) if len(w) > 2]
    if nums:
        a_nums = set(re.findall(r"\d+(?:\.\d+)?%?", a))
        if not all(n in a_nums or n.rstrip("%") in a_nums for n in nums):
            return False
        return all(w in a for w in words[:2])  # keep a light check on the unit word ("ngay", "thang")
    return all(w in a for w in words)


def score(q: dict, answer: str) -> dict:
    refused = bool(REFUSAL.search(answer))
    if not q["answerable"]:
        return {"keyfact_cov": 1.0 if refused else 0.0, "keyfact_all": refused, "refused": refused}
    facts = q.get("key_facts") or []
    hits = [fact_present(f, answer) for f in facts]
    cov = sum(hits) / len(hits) if hits else 1.0
    return {"keyfact_cov": cov, "keyfact_all": all(hits), "refused": refused, "missing_facts":
            [f for f, h in zip(facts, hits) if not h]}
