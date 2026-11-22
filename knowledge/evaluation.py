import json
import math
from dataclasses import dataclass
from pathlib import Path

from .retrieval import search_knowledge


@dataclass
class CaseResult:
    question: str
    kind: str                    # "answerable" | "out_of_scope"
    rank: int | None             # প্রত্যাশিত chunk top-k-র কত নম্বরে (১ = সবার আগে)
    hit_distance: float | None   # প্রত্যাশিত chunk-এর distance
    top_distance: float | None   # সবচেয়ে কাছের (যেকোনো) chunk-এর distance
    top_label: str


def load_cases(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _matches(chunk, expect):
    return (
        chunk.source_name == expect["source"]
        and expect["section"].lower() in chunk.heading_path.lower()
    )


def evaluate(cases, k):
    results = []
    for case in cases:
        # threshold ছাড়া: আমরা সব দূরত্ব দেখতে চাই
        chunks = search_knowledge(case["question"], k=k, max_distance=math.inf)
        top = chunks[0] if chunks else None
        expect = case.get("expect")
        rank = hit_distance = None
        if expect:
            for i, chunk in enumerate(chunks, 1):
                if _matches(chunk, expect):
                    rank, hit_distance = i, chunk.distance
                    break
        results.append(CaseResult(
            question=case["question"],
            kind="answerable" if expect else "out_of_scope",
            rank=rank,
            hit_distance=hit_distance,
            top_distance=top.distance if top else None,
            top_label=top.heading_path if top else "-",
        ))
    return results


def suggest_threshold(hit_distances, out_distances):
    """ফেরত: (প্রস্তাবিত মান বা None, দুই দল আলাদা করা গেছে কি না)।"""
    if not hit_distances:
        return None, False
    in_max = max(hit_distances)
    if out_distances and min(out_distances) > in_max:
        return round((in_max + min(out_distances)) / 2, 2), True
    return round(in_max + 0.02, 2), False