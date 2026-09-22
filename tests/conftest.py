"""Fast, deterministic stand-ins for the embedding/reranker models and Ollama, so tests run without a GPU or network."""
from __future__ import annotations

import re
import zlib

import numpy as np
import pytest

STOP = set("a an and are as at be by can do does for from how i in is it of on or the to what when who will with my me".split())


def _toks(t: str):
    t = re.sub(r"^Represent this sentence for searching relevant passages: ", "", t)
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP]


class FakeEmbedder:
    max_seq_length = 512

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False, batch_size=32):
        dim = 384
        out = np.zeros((len(texts), dim), dtype="float32")
        for r, t in enumerate(texts):
            for w in _toks(t):
                out[r, zlib.crc32(w.encode()) % dim] += 1.0
        if normalize_embeddings:
            out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
        return out


class FakeCrossEncoder:
    def predict(self, pairs):
        scores = []
        for q, p in pairs:
            qt, pt = set(_toks(q)), set(_toks(p))
            scores.append(10.0 * len(qt & pt) / max(1, len(qt)) - 3.0)
        return np.array(scores)


class FakeOllama:
    """Scripted responses. `script` maps a substring of the user message to a canned reply."""

    def __init__(self, script=None, available=False):
        self.script = script or {}
        self._available = available
        self.calls = []

    def available(self, max_age=15.0):
        return self._available

    def chat(self, messages, temperature=0.0, max_tokens=400):
        self.calls.append(messages)
        combined = "\n".join(m["content"] for m in messages)
        for key, reply in self.script.items():
            if key in combined:
                return reply
        return "INSUFFICIENT_EVIDENCE"


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_cross_encoder():
    return FakeCrossEncoder()


@pytest.fixture
def project(tmp_path):
    """A minimal document tree plus a ready Settings object, isolated per test."""
    import shutil
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "data" / "documents"
    dst = tmp_path / "data" / "documents"
    shutil.copytree(src, dst)
    from isearch.config import Settings
    s = Settings.from_env(home=tmp_path)
    s.ensure_dirs()
    return s


@pytest.fixture
def assistant(project, fake_embedder, fake_cross_encoder):
    from isearch.agent import Assistant
    from isearch.store import Store
    return Assistant(project, embedder=fake_embedder, cross_encoder=fake_cross_encoder,
                     llm=FakeOllama(available=False), store=Store(project.db_path))
