# Institutional AI Search

A hybrid, citation-checked RAG assistant for institute documents: academic regulations, admissions, hostel rules, research policy, faculty profiles, and internal circulars. Runs locally; the LLM step is optional (Ollama) and everything works without it.

This is a from-scratch rebuild of an earlier single-notebook prototype, restructured as an installable package with an API, a chat UI, tests, document versioning, and role-based access. The agent roles (router, retriever, grader, synthesizer, verifier, abstain) follow the [agentic_rag](https://github.com/patchy631/ai-engineering-hub/tree/main/agentic_rag) project in the AI Engineering Hub, implemented here as plain functions rather than CrewAI agents.

## How a question is answered

```
Question (+ conversation history)
  │
  ├─ Memory ─────────► resolve follow-ups ("and for PhD students?") into a standalone question
  ├─ Router ──────────► picks a document category, or searches everything
  ├─ Retriever ───────► BM25 + dense search fused with Reciprocal Rank Fusion, then a cross-encoder reranker
  │                     (filtered by the caller's role and by document version — superseded docs are excluded)
  ├─ Grader ──────────► drops weak passages; if nothing is relevant, widens the search, then rewrites the
  │                     query and retries
  ├─ Synthesizer ─────► local LLM answers from numbered evidence only, or an extractive answer with no LLM
  ├─ Verifier ────────► checks citations and every number/date against the evidence; one retry, then falls
  │                     back to the extractive answer
  └─ Abstain ─────────► "not found in the indexed documents" + closest matches, logged for the admin report
```

## What's new compared to the notebook prototype

| Area | Before | Now |
|---|---|---|
| Shape | One notebook, module-level globals | Installable package (`isearch`), CLI, FastAPI service, Streamlit app |
| Access | Everyone sees everything | `public` / `student` / `staff` audience levels, enforced in retrieval |
| Versions | None — an old and a new regulation could both be cited | Documents with the same base name are grouped into a family; only the newest `effective` date is searched by default |
| Conversation | Single-turn only | Follow-up questions are resolved into standalone queries using recent turns |
| Feedback loop | None | Every question is logged; thumbs-down and "couldn't find it" questions surface in `isearch report` |
| Uploads | Manual file copy + notebook re-run | `/documents` API endpoint and a Streamlit tab, admin-key protected |
| Tests | None | pytest suite (40 tests) covering chunking, ingestion, the agent loop, verification, and the API |

## Project layout

```
src/isearch/     the package: config, ingest, chunking, index, agent, verify, llm, store, service, cli
app/             streamlit_app.py — the chat UI
data/documents/  your documents, one subfolder per category (sample data included)
eval/            eval_set.json — retrieval/answer quality questions
tests/           pytest suite, with fast fake models so it runs without a GPU
```

## Setup

```bash
pip install -e ".[dev]"
python -m nltk.downloader punkt   # first run only
```

Optional: install [Ollama](https://ollama.com) and run `ollama pull llama3.2` to enable LLM answers. Without it, answers are extractive — every sentence is quoted straight from a document.

## Use it

```bash
isearch reload                                     # ingest data/documents and show what was indexed
isearch ask "Until when can I drop a course?" --role student
isearch serve                                       # FastAPI on :8000  (docs at /docs)
isearch ui                                          # Streamlit chat app on :8501
isearch eval                                        # retrieval quality, answer hit rate, threshold calibration
isearch report                                       # usage stats, unanswered questions, downvoted answers
```

`ISEARCH_HOME` (or `--home`) sets the project directory holding `data/` and `artifacts/`; defaults to the current directory. Any setting in `src/isearch/config.py` can be overridden with an `ISEARCH_<NAME>` environment variable — see `.env.example`.

## Add your documents

Put files under `data/documents/<category>/` (pdf, txt, md, or html). A `.txt`/`.md` file can start with front matter:

```
title: Academic Regulations 2024
category: academic
audience: student
effective: 2024-07-01

Course Registration
...
```

For PDF/HTML, or to override the guess, add a sidecar `yourfile.pdf.meta.json`:
```json
{"category": "academic", "audience": "staff", "effective": "2024-07-01"}
```

Files sharing a base name that differ by a date or version (`regs_2021.txt`, `regs_2024.txt`, `regs_v2.pdf`) are grouped into one family; the notebook/service treats the one with the latest `effective` date as current and hides the rest by default (`include_superseded=True` brings them back).

The bundled sample documents are **synthetic** — replace them before drawing any real conclusions. `WRITE_SAMPLES`-style logic isn't in this version: real files simply take priority over `sample_*` files whenever both exist in a folder.

## Evaluate and calibrate

`eval/eval_set.json` holds labelled questions (`contains` = a phrase the right passage must have) and a list of out-of-scope questions. `isearch eval` reports:
- Recall@5 and MRR for BM25-only, dense-only, hybrid, and hybrid+rerank
- End-to-end answer hit rate and abstention correctness (extractive mode, so it doesn't depend on Ollama)
- A suggested `RELEVANCE_THRESHOLD` from the gap between in-scope and out-of-scope reranker scores

Replace the sample questions with 20–50 real ones from your users once real documents are loaded, and re-run.

## Run the tests

```bash
pytest
```

The suite injects small deterministic stand-ins for the embedding and reranker models (`tests/conftest.py`) so it runs in seconds without downloading anything, plus a scriptable fake Ollama client to exercise the LLM/verifier/fallback paths. It covers chunking edge cases (headings, overlap, a single 500-word sentence), PDF/HTML ingestion (running headers stripped, scanned PDFs skipped not crashed, nav/script stripped, tables kept), the full agent loop (abstain, widen, rewrite, role filtering, version filtering, conversation memory), the verifier (rejects a right-looking-but-wrong number like "week 12" when the evidence says "week 8" but mentions 12 elsewhere), and every API endpoint.

## Known limits

- **Scanned PDFs** have no text layer and are skipped; run OCR (e.g. `ocrmypdf`) first.
- **The verifier checks digits, not spelled-out numbers.** "Thirty days" passes the number check even if wrong; the optional LLM cross-check catches some of these.
- **Multi-part questions** ("what's the fee and the deadline?") get one retrieval pass; splitting into sub-questions is a natural next step.
- **Large corpora**: swap `IndexFlatIP` for `IndexHNSWFlat` in `index.py` past roughly 100k chunks.
- I could not reach Hugging Face from the environment I built this in, so `sentence-transformers` and Ollama were exercised only through the fixtures in `tests/conftest.py`, not the real models. Run `isearch eval` yourself after installing to confirm real-world retrieval quality and to calibrate `RELEVANCE_THRESHOLD`.
