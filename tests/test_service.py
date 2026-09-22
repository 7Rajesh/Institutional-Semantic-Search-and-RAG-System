import pytest
from fastapi.testclient import TestClient

from isearch.agent import Assistant
from isearch.service import build_app
from isearch.store import Store
from tests.conftest import FakeOllama


@pytest.fixture
def client(project, fake_embedder, fake_cross_encoder):
    """build_app() constructs its own Assistant; swap in the fake-model Assistant for the test."""
    project.admin_key = "test-key"
    a = Assistant(project, embedder=fake_embedder, cross_encoder=fake_cross_encoder,
                 llm=FakeOllama(available=False), store=Store(project.db_path))
    import isearch.service as svc
    orig = svc.Assistant
    svc.Assistant = lambda settings: a
    try:
        app = svc.build_app(project)
    finally:
        svc.Assistant = orig
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["documents"] > 0


def test_ask_and_feedback_roundtrip(client):
    r = client.post("/ask", json={"question": "Until when can I drop a course?", "role": "staff", "use_llm": False})
    assert r.status_code == 200
    body = r.json()
    assert body["answered"] and body["query_id"] is not None
    fb = client.post("/feedback", json={"query_id": body["query_id"], "rating": 1})
    assert fb.status_code == 200


def test_search_endpoint(client):
    r = client.post("/search", json={"query": "hostel gate timing", "role": "staff"})
    assert r.status_code == 200 and len(r.json()["results"]) > 0


def test_documents_listing(client):
    r = client.get("/documents")
    assert r.status_code == 200 and len(r.json()["documents"]) > 0


def test_admin_endpoints_require_key(client):
    assert client.get("/admin/stats").status_code == 401
    assert client.get("/admin/stats", headers={"X-API-Key": "test-key"}).status_code == 200
    assert client.post("/reload", headers={"X-API-Key": "wrong"}).status_code == 401


def test_upload_requires_admin_key(client):
    files = {"file": ("new.txt", b"title: New\ncategory: policy\n\nSome rule text here.", "text/plain")}
    r = client.post("/documents?category=policy", files=files)
    assert r.status_code == 401
    r2 = client.post("/documents?category=policy", files=files, headers={"X-API-Key": "test-key"})
    assert r2.status_code == 200
