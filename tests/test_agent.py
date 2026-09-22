from isearch.agent import Assistant
from isearch.store import Store
from tests.conftest import FakeOllama


def test_abstains_on_out_of_scope_question(assistant):
    r = assistant.ask("How do I bake a chocolate cake?", role="staff", use_llm=False)
    assert not r.answered and r.mode == "abstained"


def test_extractive_answer_when_no_llm(assistant):
    r = assistant.ask("Until when can I drop a course?", role="staff", use_llm=False)
    assert r.answered and r.mode == "extractive"
    assert "week 8" in r.answer


def test_router_widens_when_forced_to_the_wrong_category(assistant, monkeypatch):
    monkeypatch.setattr(assistant, "_route", lambda q, base: (["faculty"], [("faculty", 0.5), ("hostel", 0.1)]))
    r = assistant.ask("Until when can I drop a course?", role="staff", use_llm=False)
    assert any("widened" in t for t in r.trace)
    assert r.answered


def test_forced_category_abstains_without_widening(assistant):
    r = assistant.ask("What is the tuition fee?", category="faculty", role="staff", use_llm=False)
    assert not r.answered
    assert not any("widened" in t for t in r.trace)


def test_role_hides_staff_only_documents(assistant):
    q = "What does the internal pay matrix assign to assistant professors under the staff salary bands?"
    student_hits = assistant.search(q, role="student", top_k=10)
    staff_hits = assistant.search(q, role="staff", top_k=10)
    assert not any("salary_bands" in h["source"] for h in student_hits)
    assert any("salary_bands" in h["source"] for h in staff_hits)

    r_staff = assistant.ask(q, category="policy", role="staff", use_llm=False)
    assert r_staff.answered and "Pay Level 10" in r_staff.answer
    r_student = assistant.ask(q, category="policy", role="student", use_llm=False)
    assert not any("salary_bands" in e["source"] for e in r_student.evidence)


def test_superseded_documents_are_excluded_by_default(assistant):
    r = assistant.ask("Until when can I withdraw from a course?", role="staff", use_llm=False)
    assert r.answered
    assert all(not e["superseded"] for e in r.evidence)
    assert "week 8" in r.answer and "week 6" not in r.answer


def test_conversation_memory_resolves_a_follow_up(assistant):
    first = assistant.ask("What are the hostel gate timings?", role="staff", use_llm=False)
    assert first.answered
    history = [{"q": first.query, "a": first.answer}]
    follow_up = assistant.ask("and what about visitors?", history=history, role="staff", use_llm=False)
    assert follow_up.standalone_query != "and what about visitors?"
    assert follow_up.answered


def test_llm_path_returns_cited_answer(project, fake_embedder, fake_cross_encoder):
    llm = FakeOllama(available=True, script={
        "Rewrite the question": "course withdrawal deadline",
        "Is every factual claim": "SUPPORTED.",
        "You answer questions": "A student may withdraw from a course until the end of week 8 [1].",
    })
    a = Assistant(project, embedder=fake_embedder, cross_encoder=fake_cross_encoder, llm=llm, store=Store(project.db_path))
    r = a.ask("Until when can I drop a course?", role="staff")
    assert r.mode.startswith("LLM:") and r.answered
    assert "[1]" in r.answer


def test_llm_bad_number_falls_back_to_extractive(project, fake_embedder, fake_cross_encoder):
    llm = FakeOllama(available=True, script={
        "Rewrite the question": "course withdrawal deadline",
        "You answer questions": "A student may withdraw until the end of week 12 [1].",   # wrong
    })
    a = Assistant(project, embedder=fake_embedder, cross_encoder=fake_cross_encoder, llm=llm, store=Store(project.db_path))
    r = a.ask("Until when can I drop a course?", role="staff")
    assert r.mode == "extractive (fallback after failed check)"
    assert "week 8" in r.answer


def test_llm_insufficient_evidence_triggers_abstain(project, fake_embedder, fake_cross_encoder):
    llm = FakeOllama(available=True, script={"You answer questions": "INSUFFICIENT_EVIDENCE"})
    a = Assistant(project, embedder=fake_embedder, cross_encoder=fake_cross_encoder, llm=llm, store=Store(project.db_path))
    r = a.ask("Until when can I drop a course?", role="staff")
    assert not r.answered and r.mode == "abstained"


def test_query_logged_and_feedback_recorded(assistant):
    r = assistant.ask("Until when can I drop a course?", role="staff", use_llm=False)
    assert r.query_id is not None
    assistant.store.add_feedback(r.query_id, -1, "too short")
    down = assistant.store.downvoted()
    assert down and down[0]["query_id" if "query_id" in down[0] else "id"]


def test_upload_and_delete_document(assistant):
    info = assistant.save_document("new_policy.txt", b"title: New Policy\ncategory: policy\n\nA rule about parking on campus.", "policy")
    assert info["n_chunks"] >= 1
    r = assistant.ask("What is the rule about parking?", role="staff", use_llm=False)
    assert r.answered
    assistant.remove_document(info["path"])
    r2 = assistant.ask("What is the rule about parking?", role="staff", use_llm=False)
    assert not r2.answered


def test_upload_rejects_bad_category_and_type(assistant):
    import pytest
    with pytest.raises(ValueError):
        assistant.save_document("x.exe", b"data", "policy")
    with pytest.raises(ValueError):
        assistant.save_document("x.txt", b"data", "bad category!")
