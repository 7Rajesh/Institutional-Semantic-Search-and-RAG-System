"""Offline evaluation: retrieval quality, end-to-end answer hit rate, abstention, and threshold calibration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .index import Filters

RETRIEVERS = ["BM25 only", "Dense only", "Hybrid (RRF)", "Hybrid + rerank"]


def load_eval_set(path) -> dict:
    """JSON: {"questions": [{"q":..., "contains":..., "category":optional}], "out_of_scope": ["...", ...]}"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _retrieve(assistant, name: str, q: str):
    f = Filters((), "staff", False)
    if assistant.index is None:
        return []
    if name == "BM25 only":
        return assistant.index.search(q, f, mode="bm25")
    if name == "Dense only":
        return assistant.index.search(q, f, mode="dense")
    if name == "Hybrid (RRF)":
        return assistant.index.search(q, f)
    return assistant._rerank(q, assistant.index.search(q, f))


def evaluate_retrieval(assistant, items: list[dict], k: int = 5) -> list[dict]:
    rows = []
    for name in RETRIEVERS:
        hits, rr = 0, []
        for item in items:
            ranked = _retrieve(assistant, name, item["q"])[:k]
            pos = next((r for r, c in enumerate(ranked, 1)
                        if item["contains"].lower() in c["chunk"]["text"].lower()), None)
            hits += pos is not None
            rr.append(1 / pos if pos else 0.0)
        rows.append({"retriever": name, f"recall@{k}": round(hits / max(1, len(items)), 3),
                     "mrr": round(float(np.mean(rr)) if rr else 0.0, 3)})
    return rows


def evaluate_answers(assistant, eval_set: dict) -> dict:
    """End to end, without the LLM: does the answer contain the fact, and does it abstain when it should?"""
    items, oos = eval_set["questions"], eval_set.get("out_of_scope", [])
    correct, wrongly_abstained, misses = 0, 0, []
    for item in items:
        a = assistant.ask(item["q"], role="staff", use_llm=False, log=False)
        ok = a.answered and item["contains"].lower() in a.answer.lower()
        correct += ok
        wrongly_abstained += not a.answered
        if not ok:
            misses.append(item["q"])
    wrongly_answered = [q for q in oos if assistant.ask(q, role="staff", use_llm=False, log=False).answered]
    return {
        "answer_hit_rate": round(correct / max(1, len(items)), 3),
        "wrongly_abstained": wrongly_abstained,
        "abstains_on_out_of_scope": round(1 - len(wrongly_answered) / max(1, len(oos)), 3),
        "wrongly_answered_out_of_scope": wrongly_answered,
        "missed_questions": misses,
    }


def calibrate(assistant, eval_set: dict) -> dict:
    """Compare the best reranker score of in-scope vs. out-of-scope questions to suggest a threshold."""
    def best(q):
        ranked = _retrieve(assistant, "Hybrid + rerank", q)
        return ranked[0]["rerank"] if ranked else float("-inf")

    inside = [best(i["q"]) for i in eval_set["questions"]]
    outside = [best(q) for q in eval_set.get("out_of_scope", [])]
    result = {"in_scope_min": min(inside), "in_scope_median": float(np.median(inside)),
              "out_of_scope_max": max(outside) if outside else None,
              "current_threshold": assistant.s.relevance_threshold, "suggested_threshold": None}
    if outside and min(inside) > max(outside):
        result["suggested_threshold"] = round((min(inside) + max(outside)) / 2, 2)
    return result
