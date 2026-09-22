# Institutional AI Semantic Search and RAG system

A question-answering assistant for institute documents: academic regulations, admissions, hostel rules, research policy, faculty pages, internal circulars. Ask it in plain language and it finds the right passage, answers with a citation, and tells you when it can't find anything rather than guessing.

Everything runs locally. The LLM step is optional (Ollama) — without it you still get answers, just extractive ones, built from sentences pulled straight out of the source documents.

## How it answers a question

A question goes through a short pipeline instead of a single similarity search:

1. **Memory** – if it's a follow-up ("and for PhD students?"), fold it into a standalone question using the last few turns.
2. **Router** – guess which document category the question belongs to, or search everything if it's unclear.
3. **Retrieve** – BM25 and dense embeddings run in parallel, fused with Reciprocal Rank Fusion, then reranked with a cross-encoder. Filtered by the asker's role and by document version, so a superseded policy doesn't get cited as current.
4. **Grade** – drop weak matches. If nothing clears the bar, widen the category, then rewrite the query and try again.
5. **Answer** – the local LLM writes a short answer from the numbered evidence, or, without an LLM, the top matching sentences are returned as-is.
6. **Verify** – every citation and every number in the answer gets checked against the evidence. If something doesn't match, it retries once, then falls back to the extractive answer.
7. **Abstain** – if nothing relevant turns up, it says so and shows the closest passages instead of making something up.

The role split (router, retriever, grader, synthesizer, verifier) borrows from the [agentic_rag](https://github.com/patchy631/ai-engineering-hub/tree/main/agentic_rag) project in the AI Engineering Hub. Here it's plain Python functions instead of CrewAI agents, which makes it easier to read and to debug when something goes wrong.



- **Old documents keep getting cited as current.** Files sharing a name but differing by year or version (`regs_2021.txt`, `regs_2024.txt`) are grouped into a family. Only the newest `effective` date is searched by default.
- **Not everyone should see everything.** Salary bands and internal circulars shouldn't show up in a student's search. Each document has a `public` / `student` / `staff` level, enforced in retrieval, not just hidden in the UI.
- **You need to know what it's failing at.** Every question gets logged, along with whether it was answered and any thumbs-down feedback. `isearch report` surfaces the questions it couldn't answer, which is the actual signal for where the documents have gaps.
- **A number that's merely present isn't a number that's correct.** The verifier checks that a number in the answer sits next to the same neighbouring words in the evidence, not just that the digits appear somewhere in the passage — catching things like "week 12" when the source says "week 8" but happens to mention 12 credits nearby.

## Project layout

```
src/isearch/     the package — config, ingestion, chunking, index, agent, verifier, LLM client, store, API, CLI
app/             streamlit_app.py, the chat UI
data/documents/  your documents, one folder per category (sample data included)
eval/            eval_set.json — labelled questions for measuring retrieval and answer quality
tests/           pytest suite, using small fake models so it runs in a couple of seconds
```

## Setting it up

```bash
pip install -e ".[dev]"
python -m nltk.downloader punkt
```

Optional: install [Ollama](https://ollama.com) and run `ollama pull llama3.2` to get prose answers instead of extractive ones.

## Running it

```bash
isearch reload                                   # index data/documents and print what it found
isearch ask "Until when can I drop a course?" --role student
isearch serve                                     # FastAPI on :8000
isearch ui                                        # Streamlit chat app on :8501
isearch eval                                       # recall/MRR, answer hit rate, threshold calibration
isearch report                                     # usage stats, unanswered questions, downvotes
```

`--home` (or the `ISEARCH_HOME` env var) points at the project directory holding `data/` and `artifacts/`. Anything in `config.py` can be overridden with an `ISEARCH_<NAME>` environment variable — see `.env.example`.

## Adding your own documents

Drop pdf, txt, md, or html files into `data/documents/<category>/`. A text or markdown file can carry front matter:

```
title: Academic Regulations 2024
category: academic
audience: student
effective: 2024-07-01

Course Registration
...
```

For PDFs and HTML, add a sidecar file instead — `yourfile.pdf.meta.json`:

```json
{"category": "academic", "audience": "staff", "effective": "2024-07-01"}
```

Real files always take priority over the bundled `sample_*` files, so you don't need to delete the samples by hand — just add your own documents and they'll be used instead.

The sample documents are made up for testing. Every number and rule in them is invented, so don't treat them as real policy.

## Checking retrieval quality

`eval/eval_set.json` has labelled questions (a phrase the right passage should contain) plus a handful of out-of-scope ones. `isearch eval` reports recall@5 and MRR for BM25-only, dense-only, hybrid, and hybrid+rerank, an end-to-end answer hit rate, and a suggested relevance threshold based on the gap between in-scope and out-of-scope scores.

The sample eval set is small and will look artificially perfect. Swap in 20–50 real questions once real documents are loaded and re-run before trusting the numbers.

## Tests

```bash
pytest
```

The suite swaps in small deterministic stand-ins for the embedding and reranker models, so it doesn't need a GPU or a download to run. It covers chunking edge cases, PDF and HTML ingestion, the full agent loop (abstaining, widening, rewriting, role and version filtering, conversation memory), the verifier, and the API endpoints.

##  limitations

- Scanned PDFs have no text layer and get skipped, not crashed on. Run them through OCR first.
- The verifier checks digits, not words, so "thirty days" would slip past even if the source says something else. The optional LLM cross-check catches some of this but not all of it.
- A question with two parts ("what's the fee and the deadline?") only gets one retrieval pass. Splitting it into sub-questions would be the natural next step.
- Past roughly 100k chunks, swap `IndexFlatIP` for `IndexHNSWFlat` in `index.py`.
- I built and tested this without access to Hugging Face or a real Ollama instance, so the retrieval quality numbers above are unverified. Run `isearch eval` yourself after installing before relying on the threshold it picks.
