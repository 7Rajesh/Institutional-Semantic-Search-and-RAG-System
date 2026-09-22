"""The assistant: router -> retriever -> grader -> (rewrite/retry) -> synthesizer -> verifier, with conversation memory."""
from __future__ import annotations

import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Settings
from .index import Filters, HybridIndex
from .ingest import SUPPORTED_SUFFIXES, DocInfo, ingest_directory
from .llm import OllamaClient
from .models import load_cross_encoder, load_embedder
from .store import Store
from .text import split_sentences, stem
from .verify import check_answer

# Words students use vs. words the documents use. Extend for your institute (acronyms, office names).
SYNONYMS = {
    "drop": ["withdraw", "withdrawal"], "quit": ["withdraw"], "fees": ["fee", "tuition"],
    "hostel": ["residence"], "leave": ["absence"], "guest": ["visitor"], "friend": ["visitor"],
    "marks": ["grade", "cgpa"], "phd": ["doctoral", "thesis"], "professor": ["faculty"],
    "exam": ["examination"], "fine": ["penalty", "late fee"], "book": ["library"],
}

SYSTEM_PROMPT = """You answer questions for students and staff of an institute using ONLY the numbered evidence passages.
Rules:
1. End every factual sentence with its citation, for example [1] or [1][3].
2. Copy dates, numbers, amounts and deadlines exactly as written in the evidence.
3. If the evidence does not answer the question, reply with exactly: INSUFFICIENT_EVIDENCE
4. If passages disagree, prefer the one with the latest effective date and say the other is superseded. Cite both.
5. Never use outside knowledge. Keep the answer under 150 words."""

_PRONOUNS = {"it", "that", "this", "they", "them", "those", "these", "there", "he", "she", "its"}


@dataclass
class Answer:
    query: str
    standalone_query: str
    answer: str
    mode: str                       # "LLM: <model>" | "extractive" | "extractive (fallback...)" | "abstained"
    answered: bool
    evidence: list[dict] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    best_score: float | None = None
    latency_ms: float = 0.0
    query_id: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Assistant:
    def __init__(self, settings: Settings | None = None, *, embedder=None, cross_encoder=None, llm=None, store=None):
        self.s = settings or Settings.from_env()
        self.s.ensure_dirs()
        self.embedder = embedder if embedder is not None else load_embedder(self.s)
        self.cross_encoder = cross_encoder if cross_encoder is not None else load_cross_encoder(self.s)
        self.llm = llm if llm is not None else OllamaClient(self.s)
        self.store = store if store is not None else Store(self.s.db_path)
        self._lock = threading.RLock()
        self.index: HybridIndex | None = None
        self.docs: list[DocInfo] = []
        self.reload()

    # ------------------------------------------------------------ documents
    def reload(self) -> None:
        """Re-read the document folder and rebuild the indexes. Only changed text is re-embedded."""
        result = ingest_directory(self.s)
        index = HybridIndex(result.chunks, self.embedder, self.s) if result.chunks else None
        with self._lock:
            self.index, self.docs = index, result.docs

    def documents(self) -> list[dict]:
        return [d.to_dict() for d in self.docs]

    def save_document(self, filename: str, data: bytes, category: str = "general") -> dict:
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)
        if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported file type; use one of {sorted(SUPPORTED_SUFFIXES)}")
        if not re.fullmatch(r"[a-z0-9_-]{1,40}", category):
            raise ValueError("category must be 1-40 characters: lowercase letters, digits, - or _")
        if len(data) > self.s.max_upload_mb * 1024 * 1024:
            raise ValueError(f"file is larger than {self.s.max_upload_mb} MB")
        if name.startswith("sample_"):
            name = "doc_" + name                              # sample_* files are ignored by ingestion
        dest = self.s.document_dir / category / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        self.reload()
        rel = f"{category}/{name}"
        return next(d.to_dict() for d in self.docs if d.path == rel)

    def remove_document(self, rel_path: str) -> None:
        root = self.s.document_dir.resolve()
        target = (root / rel_path).resolve()
        if root not in target.parents or not target.is_file():
            raise FileNotFoundError(rel_path)
        target.unlink()
        side = target.with_name(target.name + ".meta.json")
        if side.exists():
            side.unlink()
        self.reload()

    # ------------------------------------------------------------ retrieval
    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        candidates = candidates[: self.s.rerank_input_k]
        if not candidates:
            return []
        scores = self.cross_encoder.predict([(query, c["chunk"]["context_text"]) for c in candidates])
        out = [{**c, "rerank": float(s)} for c, s in zip(candidates, scores)]
        return sorted(out, key=lambda r: r["rerank"], reverse=True)

    def _retrieve(self, query: str, filters: Filters) -> list[dict]:
        index = self.index
        return self._rerank(query, index.search(query, filters)) if index else []

    def search(self, query: str, *, category: str | None = None, role: str = "student",
               include_superseded: bool = False, top_k: int | None = None) -> list[dict]:
        """Retrieval only (no answer generation)."""
        f = Filters((category,) if category else (), role, include_superseded)
        ranked = self._retrieve(query, f)[: top_k or self.s.final_top_k]
        return [self._evidence(i, r) for i, r in enumerate(ranked, start=1)]

    @staticmethod
    def _evidence(n: int, r: dict) -> dict:
        c = r["chunk"]
        return {"n": n, "source": c["source"], "title": c["title"], "page": c["page"], "section": c["section"],
                "category": c["category"], "effective": c["effective"], "superseded": c["superseded"],
                "text": c["text"], "score": round(r["rerank"], 3)}

    def _grade(self, ranked: list[dict]) -> dict:
        if not ranked:
            return {"passed": False, "best": float("-inf"), "evidence": []}
        best = ranked[0]["rerank"]
        kept = [r for r in ranked if r["rerank"] >= self.s.relevance_threshold
                and r["rerank"] >= best - self.s.relevance_window][: self.s.final_top_k]
        return {"passed": bool(kept), "best": best, "evidence": kept}

    def _route(self, query: str, base: Filters):
        scores = self.index.category_scores(query, base)
        if len(scores) < 2:
            return None, scores
        (top, s1), (_, s2) = scores[0], scores[1]
        return ([top] if s1 - s2 >= self.s.route_margin else None), scores

    # ------------------------------------------------------------ language steps
    def _condense(self, question: str, history: list[dict], use_llm: bool, trace: list[str]) -> str:
        """Turn a follow-up ("and for PhD students?") into a standalone question."""
        if not history:
            return question
        turns = history[-self.s.history_turns:]
        if use_llm:
            try:
                convo = "\n".join(f"User: {t['q']}\nAssistant: {t['a'][:300]}" for t in turns)
                prompt = ("Rewrite the last user question as a standalone question, using the conversation for "
                          "context. If it is already standalone, repeat it. Output only the question.\n\n"
                          f"{convo}\nUser: {question}")
                out = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=60).strip().strip('"')
                if out:
                    if out != question:
                        trace.append(f"Memory: follow-up rewritten as {out!r}")
                    return out
            except Exception:
                pass
        words = re.findall(r"[a-z0-9]+", question.lower())
        if len(words) <= 6 or _PRONOUNS & set(words):
            standalone = f"{turns[-1]['q']} {question}"
            trace.append(f"Memory: follow-up combined with the previous question -> {standalone!r}")
            return standalone
        return question

    def _rewrite(self, query: str, use_llm: bool) -> str:
        if use_llm:
            try:
                prompt = ("Rewrite the question as a short keyword search query for institute policy documents. "
                          "Use formal terms such as 'withdrawal' instead of 'drop'. Output only the query.\n\n"
                          f"Question: {query}")
                out = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=40).strip().strip('"')
                if out:
                    return out
            except Exception:
                pass
        extra = []
        for tok in re.findall(r"[a-z0-9]+", query.lower()):
            extra += SYNONYMS.get(tok, []) + SYNONYMS.get(stem(tok), [])
        return (query + " " + " ".join(dict.fromkeys(extra))).strip()

    @staticmethod
    def _format_evidence(evidence: list[dict]) -> str:
        blocks = []
        for i, r in enumerate(evidence, start=1):
            c = r["chunk"]
            page = f", p.{c['page']}" if c["page"] else ""
            eff = f" | Effective: {c['effective']}" if c["effective"] else ""
            old = " (SUPERSEDED)" if c["superseded"] else ""
            blocks.append(f"[{i}] Source: {c['source']}{page} | Section: {c['section'] or 'n/a'}{eff}{old}\n{c['text']}")
        return "\n\n".join(blocks)

    def _extractive(self, query: str, evidence: list[dict], max_sentences: int = 3) -> str:
        candidates = [(n, s) for n, r in enumerate(evidence, start=1)
                      for s in split_sentences(r["chunk"]["text"]) if len(s.split()) >= 4]
        if not candidates:
            return ""
        scores = self.cross_encoder.predict([(query, s) for _, s in candidates])
        ranked = sorted(zip(scores, range(len(candidates))), reverse=True)
        picked = sorted(i for k, (sc, i) in enumerate(ranked[:max_sentences])
                        if k == 0 or sc >= self.s.relevance_threshold)
        return "\n".join(f"- {candidates[i][1]} [{candidates[i][0]}]" for i in picked)

    def _llm_answer(self, query: str, evidence: list[dict], trace: list[str]):
        """Returns (status, text); status is 'ok', 'insufficient' or 'failed'."""
        feedback = None
        texts = [r["chunk"]["text"] for r in evidence]
        for attempt in (1, 2):
            user = f"Evidence:\n{self._format_evidence(evidence)}\n\nQuestion: {query}"
            if feedback:
                user += f"\n\nYour previous answer was rejected: {feedback}. Answer again and fix this."
            try:
                draft = self.llm.chat([{"role": "system", "content": SYSTEM_PROMPT},
                                       {"role": "user", "content": user}], max_tokens=350)
            except Exception as exc:
                trace.append(f"Synthesizer: LLM call failed ({exc})")
                return "failed", None
            if "INSUFFICIENT_EVIDENCE" in draft:
                trace.append("Synthesizer: model says the evidence does not answer the question")
                return "insufficient", None
            problems = check_answer(draft, texts)
            if not problems and self.s.llm_verify:
                try:
                    verdict = self.llm.chat([{"role": "user", "content":
                        f"Evidence:\n{self._format_evidence(evidence)}\n\nAnswer:\n{draft}\n\n"
                        "Is every factual claim in the answer supported by the evidence? "
                        "Reply SUPPORTED or UNSUPPORTED, then one short reason."}], max_tokens=60)
                    if not verdict.strip().upper().startswith("SUPPORTED"):
                        problems.append(f"LLM check: {verdict}")
                except Exception:
                    pass
            trace.append(f"Verifier (draft {attempt}): " + ("passed" if not problems else "; ".join(problems)))
            if not problems:
                return "ok", draft
            feedback = "; ".join(problems)
        return "failed", None

    # ------------------------------------------------------------ the agent loop
    def ask(self, question: str, *, history: list[dict] | None = None, session_id: str | None = None,
            category: str | None = None, role: str = "student", use_llm: bool | None = None,
            include_superseded: bool = False, log: bool = True) -> Answer:
        t0 = time.perf_counter()
        trace: list[str] = []
        llm_ok = self.llm.available()
        use_llm = llm_ok if use_llm is None else (use_llm and llm_ok)

        def finish(text, mode, answered, evidence=(), best=None) -> Answer:
            ans = Answer(query=question, standalone_query=standalone, answer=text, mode=mode, answered=answered,
                         evidence=[self._evidence(i, r) for i, r in enumerate(evidence, start=1)],
                         trace=trace, best_score=best, latency_ms=round((time.perf_counter() - t0) * 1000, 1))
            if log and self.store:
                ans.query_id = self.store.log_query(
                    session_id=session_id, question=question, standalone=standalone, mode=mode, answered=answered,
                    best_score=best, latency_ms=ans.latency_ms, sources=[e["source"] for e in ans.evidence], answer=text)
            return ans

        standalone = question
        if self.index is None:
            trace.append("No documents are indexed")
            return finish("No documents are indexed yet. Add files to the documents folder.", "abstained", False)

        standalone = self._condense(question, history or [], use_llm, trace)
        base = Filters((), role, include_superseded)

        # 1. route
        if category:
            routed = [category]
            trace.append(f"Router: category fixed to '{category}' by the caller")
        else:
            routed, scores = self._route(standalone, base)
            top = ", ".join(f"{c} {s:.2f}" for c, s in scores[:3])
            trace.append(f"Router: {'search ' + routed[0] if routed else 'no clear category, search all'}  [{top}]")

        # 2. plan: routed -> widen -> rewrite
        plan = [(standalone, routed, "search")]
        if routed and not category:
            plan.append((standalone, None, "widened to all categories"))
        plan.append((None, routed if category else None, "rewritten query"))

        grade, last_ranked = None, []
        for attempt, (q, cats, label) in enumerate(plan[: self.s.max_attempts], start=1):
            if q is None:
                q = self._rewrite(standalone, use_llm)
                if q.strip().lower() == standalone.strip().lower():
                    trace.append(f"Attempt {attempt} ({label}): rewriter found nothing to change, skipped")
                    continue
            ranked = self._retrieve(q, base.with_categories(cats))
            grade = self._grade(ranked)
            last_ranked = ranked or last_ranked
            trace.append(f"Attempt {attempt} ({label}): query={q!r} categories={cats or 'all'} "
                         f"best score={grade['best']:.2f} -> {'PASS' if grade['passed'] else 'FAIL'}")
            if grade["passed"]:
                break

        # 3. abstain
        def abstain(reason: str) -> Answer:
            closest = "\n".join(f"- `{r['chunk']['source']}`, {r['chunk']['section'] or 'n/a'} (score {r['rerank']:.1f})"
                                for r in last_ranked[:3])
            text = f"{reason} I will not guess. Please check with the relevant office."
            if closest:
                text += f"\n\nClosest passages found:\n{closest}"
            return finish(text, "abstained", False, (), grade["best"] if grade else None)

        if not grade or not grade["passed"]:
            return abstain("I could not find this in the indexed documents.")
        evidence, best = grade["evidence"], grade["best"]

        # 4. synthesize + verify
        if use_llm:
            status, text = self._llm_answer(standalone, evidence, trace)
            if status == "ok":
                return finish(text, f"LLM: {self.s.ollama_model}", True, evidence, best)
            if status == "insufficient":
                return abstain("The retrieved passages do not answer this question.")
            trace.append("Falling back to the extractive answer (it quotes the documents directly)")
            mode = "extractive (fallback after failed check)"
        else:
            mode = "extractive"
        return finish(self._extractive(standalone, evidence), mode, True, evidence, best)
