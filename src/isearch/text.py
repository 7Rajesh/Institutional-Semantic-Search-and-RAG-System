"""Text utilities shared by ingestion and retrieval."""
from __future__ import annotations

import re
import unicodedata

from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()
_stem_cache: dict[str, str] = {}

STOPWORDS = set(
    """a an and are as at be been by can could do does for from had has have how i if in into is it its
    may me my of on or our shall should so than that the their them then there these they this to us was we were what when
    where which who whom why will with would you your""".split()
)


def stem(word: str) -> str:
    if word not in _stem_cache:
        _stem_cache[word] = _stemmer.stem(word)
    return _stem_cache[word]


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)        # fixes ligatures such as "ﬁ"
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)          # rejoin words hyphenated at line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_ABBREV = re.compile(r"\b(Dr|Prof|Mr|Ms|Mrs|No|vs|etc|Fig|Sec|Rs|e\.g|i\.e)\.")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    text = _ABBREV.sub(lambda m: m.group(0).replace(".", "\u2024"), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", text)
    return [p.replace("\u2024", ".") for p in parts if p]
