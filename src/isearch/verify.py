"""Answer verification. Deterministic checks first; an optional LLM check on top."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"\d[\d,.:]*\d|\d|[a-z]+")


def _tokens(text: str) -> list[str]:
    text = re.sub(r"\[\d+\]", " ", text)                       # citation markers
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)             # list numbering
    return [t.replace(",", "") for t in _TOKEN.findall(text.lower())]


def unsupported_numbers(answer: str, evidence_text: str) -> list[str]:
    """Numbers in the answer that do not appear in the evidence next to the same neighbouring word.

    The neighbour check catches "week 12" when the evidence says "week 8" but mentions "12 credits" elsewhere.
    """
    ev = _tokens(evidence_text)
    ev_pairs = set(zip(ev, ev[1:]))
    ans = _tokens(answer)
    bad = []
    for i, tok in enumerate(ans):
        if not tok[0].isdigit():
            continue
        prev_pair = (ans[i - 1], tok) if i > 0 else None
        next_pair = (tok, ans[i + 1]) if i + 1 < len(ans) else None
        if not ((prev_pair in ev_pairs) or (next_pair in ev_pairs)):
            bad.append(" ".join(ans[max(0, i - 1): i + 2]))
    return bad


def check_answer(answer: str, evidence_texts: list[str]) -> list[str]:
    """Return a list of problems (empty means the answer passed)."""
    problems = []
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    invalid = sorted(n for n in cited if not 1 <= n <= len(evidence_texts))
    if not cited:
        problems.append("the answer has no citations")
    if invalid:
        problems.append(f"citations {invalid} do not exist")
    valid = [n for n in cited if 1 <= n <= len(evidence_texts)] or range(1, len(evidence_texts) + 1)
    bad = unsupported_numbers(answer, " ".join(evidence_texts[n - 1] for n in valid))
    if bad:
        problems.append(f"numbers not found in the cited evidence in this context: {bad}")
    return problems
