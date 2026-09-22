"""Hybrid index: BM25 + dense vectors fused with Reciprocal Rank Fusion, with access/version/category filters."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .config import Settings
from .text import tokenize

AUDIENCE_LEVEL = {"public": 0, "student": 1, "staff": 2}


@dataclass(frozen=True)
class Filters:
    categories: tuple[str, ...] = ()
    role: str = "student"                 # public < student < staff
    include_superseded: bool = False

    def with_categories(self, categories) -> "Filters":
        return Filters(tuple(categories or ()), self.role, self.include_superseded)


def reciprocal_rank_fusion(rankings, weights, k: int) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + weight / (k + rank)
    return scores


def _cache_key(model: str, text: str) -> str:
    return hashlib.sha1((model + "\x00" + text).encode()).hexdigest()


def embed_with_cache(chunks, embedder, settings: Settings) -> np.ndarray:
    """Encode only chunks whose text is new. Editing one document re-embeds only that document."""
    path = settings.artifact_dir / "embedding_cache.npz"
    cache: dict[str, np.ndarray] = {}
    if path.exists():
        z = np.load(path, allow_pickle=False)
        cache = dict(zip(z["keys"].tolist(), z["vecs"]))

    keys = [_cache_key(settings.embedding_model, c["context_text"]) for c in chunks]
    missing = [i for i, k in enumerate(keys) if k not in cache]
    if missing:
        vecs = embedder.encode(
            [chunks[i]["context_text"] for i in missing],
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False, batch_size=32,
        ).astype("float32")
        for i, v in zip(missing, vecs):
            cache[keys[i]] = v

    live = {k: cache[k] for k in keys}                      # drop vectors of deleted text
    if missing or len(live) != len(cache):
        if live:
            np.savez(path, keys=np.array(list(live)), vecs=np.stack(list(live.values())))
    return np.stack([cache[k] for k in keys]).astype("float32")


class HybridIndex:
    def __init__(self, chunks: list[dict], embedder, settings: Settings):
        if not chunks:
            raise ValueError("No indexable text found. Add documents to " + str(settings.document_dir))
        self.s = settings
        self.chunks = chunks
        self.embedder = embedder
        self.embeddings = embed_with_cache(chunks, embedder, settings)
        self.dense = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.dense.add(self.embeddings)
        self.bm25 = BM25Okapi([tokenize(c["context_text"]) for c in chunks])

        self._cat = np.array([c["category"] for c in chunks], dtype=object)
        self._aud = np.array([AUDIENCE_LEVEL.get(c["audience"], 0) for c in chunks])
        self._sup = np.array([bool(c["superseded"]) for c in chunks])
        self.categories = sorted({str(c) for c in self._cat})
        self._centroids = {}
        for cat in self.categories:
            v = self.embeddings[self._cat == cat].mean(axis=0)
            self._centroids[cat] = v / (np.linalg.norm(v) + 1e-12)

    # ---- filtering
    def mask(self, f: Filters, use_categories: bool = True) -> np.ndarray:
        m = self._aud <= AUDIENCE_LEVEL.get(f.role, 1)
        if not f.include_superseded:
            m &= ~self._sup
        if use_categories and f.categories:
            m &= np.isin(self._cat, list(f.categories))
        return m

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.embedder.encode([self.s.query_prefix + query], convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype("float32")

    # ---- single retrievers
    def bm25_ranked(self, query: str, mask: np.ndarray, k: int | None = None):
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = np.where(mask, np.asarray(self.bm25.get_scores(tokens), dtype=float), -np.inf)
        order = np.argsort(-scores)[: k or self.s.bm25_top_k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def dense_ranked(self, qvec: np.ndarray, mask: np.ndarray, k: int | None = None):
        sims, ids = self.dense.search(qvec, self.dense.ntotal)          # exact search, filter afterwards
        pairs = [(int(i), float(s)) for i, s in zip(ids[0], sims[0]) if i >= 0 and mask[i]]
        return pairs[: k or self.s.dense_top_k]

    # ---- router support
    def category_scores(self, query: str, f: Filters):
        visible = set(map(str, self._cat[self.mask(f, use_categories=False)]))
        q = self.embed_query(query)[0]
        scores = [(c, float(v @ q)) for c, v in self._centroids.items() if c in visible]
        return sorted(scores, key=lambda x: -x[1])

    # ---- fused search
    def search(self, query: str, f: Filters, mode: str = "hybrid") -> list[dict]:
        mask = self.mask(f)
        if not mask.any():
            return []
        qvec = self.embed_query(query)
        bm25_pairs = self.bm25_ranked(query, mask) if mode in {"hybrid", "bm25"} else []
        dense_pairs = self.dense_ranked(qvec, mask) if mode in {"hybrid", "dense"} else []
        fused = reciprocal_rank_fusion(
            [[i for i, _ in bm25_pairs], [i for i, _ in dense_pairs]],
            [self.s.bm25_weight, self.s.dense_weight], self.s.rrf_k,
        )
        bm25_raw = dict(bm25_pairs)
        return [
            {"idx": idx, "chunk": self.chunks[idx], "rrf": score, "bm25": bm25_raw.get(idx, 0.0),
             "dense": float(self.embeddings[idx] @ qvec[0])}
            for idx, score in sorted(fused.items(), key=lambda kv: -kv[1])
        ]
