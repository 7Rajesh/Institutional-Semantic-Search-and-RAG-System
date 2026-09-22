"""Command-line interface: `isearch ask|serve|ui|reload|eval|report`."""
from __future__ import annotations

import argparse
import json
import sys

from .config import Settings


def _assistant(args):
    from .agent import Assistant
    return Assistant(Settings.from_env(home=args.home))


def cmd_ask(args):
    a = _assistant(args)
    r = a.ask(args.question, role=args.role, category=args.category, use_llm=None if not args.no_llm else False)
    print(f"\nMode: {r.mode}  |  answered: {r.answered}  |  {r.latency_ms} ms\n")
    print(r.answer)
    if r.evidence:
        print("\nSources:")
        for e in r.evidence:
            page = f", p.{e['page']}" if e["page"] else ""
            print(f"  [{e['n']}] {e['source']}{page}  ({e['section'] or 'n/a'}, score {e['score']})")
    if args.trace:
        print("\nTrace:\n  " + "\n  ".join(r.trace))


def cmd_reload(args):
    a = _assistant(args)
    print(f"Indexed {len(a.docs)} documents.")
    for d in a.docs:
        flag = " [SKIPPED: " + d.note + "]" if d.status == "skipped" else (" [superseded]" if d.superseded else "")
        print(f"  {d.path:45s} {d.category:12s} {d.audience:8s} chunks={d.n_chunks}{flag}")


def cmd_serve(args):
    import uvicorn
    uvicorn.run("isearch.service:build_app", factory=True, host=args.host, port=args.port, reload=False)


def cmd_ui(args):
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    str(__import__("pathlib").Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"),
                    "--server.port", str(args.port)])


def cmd_eval(args):
    from .evaluate import calibrate, evaluate_answers, evaluate_retrieval, load_eval_set
    a = _assistant(args)
    eval_set = load_eval_set(args.eval_set)
    print("Retrieval quality:")
    for row in evaluate_retrieval(a, eval_set["questions"]):
        print(f"  {row['retriever']:18s} recall@5={row['recall@5']:.3f}  mrr={row['mrr']:.3f}")
    print("\nEnd-to-end (extractive):")
    print(" ", json.dumps(evaluate_answers(a, eval_set), indent=2))
    print("\nThreshold calibration:")
    print(" ", json.dumps(calibrate(a, eval_set), indent=2))


def cmd_report(args):
    a = _assistant(args)
    print("Stats:", json.dumps(a.store.stats(), indent=2))
    print("\nTop unanswered questions (documentation gaps):")
    for row in a.store.unanswered(args.limit):
        print(f"  x{row['times']:<3d} {row['question']}")
    print("\nRecently downvoted answers:")
    for row in a.store.downvoted(args.limit):
        print(f"  #{row['id']} {row['question']!r}: {row['comment'] or '(no comment)'}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="isearch", description="Institutional AI Search")
    p.add_argument("--home", default=".", help="project directory holding data/ and artifacts/")
    sub = p.add_subparsers(dest="cmd", required=True)

    ask = sub.add_parser("ask", help="ask a question from the terminal")
    ask.add_argument("question")
    ask.add_argument("--role", default="student", choices=["public", "student", "staff"])
    ask.add_argument("--category", default=None)
    ask.add_argument("--no-llm", action="store_true", help="force the extractive answer")
    ask.add_argument("--trace", action="store_true")
    ask.set_defaults(func=cmd_ask)

    reload_p = sub.add_parser("reload", help="re-ingest documents and list what was indexed")
    reload_p.set_defaults(func=cmd_reload)

    serve = sub.add_parser("serve", help="run the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    ui = sub.add_parser("ui", help="run the Streamlit chat app")
    ui.add_argument("--port", type=int, default=8501)
    ui.set_defaults(func=cmd_ui)

    ev = sub.add_parser("eval", help="run retrieval + answer evaluation and threshold calibration")
    ev.add_argument("--eval-set", default="eval/eval_set.json")
    ev.set_defaults(func=cmd_eval)

    rep = sub.add_parser("report", help="usage stats, unanswered questions, downvoted answers")
    rep.add_argument("--limit", type=int, default=20)
    rep.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
