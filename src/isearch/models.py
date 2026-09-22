"""Model loading. Imports are lazy so the package imports without torch installed."""
from __future__ import annotations

from .config import Settings


def load_embedder(settings: Settings):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding_model)
    model.max_seq_length = 512
    return model


def load_cross_encoder(settings: Settings):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model, max_length=512)
