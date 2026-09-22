import json

from isearch.evaluate import calibrate, evaluate_answers, evaluate_retrieval, load_eval_set


def test_eval_set_loads(project):
    eval_set = load_eval_set(project.home.parent.parent / "eval" / "eval_set.json") if False else None
    # Use the bundled eval set directly by path.
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "eval" / "eval_set.json"
    data = load_eval_set(path)
    assert data["questions"] and data["out_of_scope"]


def test_evaluate_retrieval_runs_for_all_retrievers(assistant):
    items = json.loads((__import__("pathlib").Path(__file__).resolve().parents[1] / "eval" / "eval_set.json").read_text())
    rows = evaluate_retrieval(assistant, items["questions"][:5])
    assert len(rows) == 4
    assert all("recall@5" in r and "mrr" in r for r in rows)


def test_evaluate_answers_and_calibrate(assistant):
    eval_set = json.loads((__import__("pathlib").Path(__file__).resolve().parents[1] / "eval" / "eval_set.json").read_text())
    result = evaluate_answers(assistant, eval_set)
    assert 0.0 <= result["answer_hit_rate"] <= 1.0
    calib = calibrate(assistant, eval_set)
    assert "suggested_threshold" in calib
