"""Structure-aware chunking: split at headings, then pack whole sentences."""
from __future__ import annotations

import re

from .text import split_sentences

_NUMBERED = re.compile(r"^(\d+(\.\d+)*[.)]?|[A-Z][.)]|(Section|Article|Clause|Chapter)\s+\d+[.:]?)\s+[A-Z]")


def looks_like_heading(line: str, prev_blank: bool, next_has_text: bool) -> bool:
    s = line.strip()
    if not s or len(s) > 100:
        return False
    if s.startswith("#"):
        return True
    words = s.split()
    if len(words) > 12 or s[-1] in ".,;:!?":
        return False
    if _NUMBERED.match(s):
        return True
    if any(c.isalpha() for c in s) and s.isupper():
        return True
    if prev_blank and next_has_text and len(words) <= 8:
        return all(w[0].isupper() or not w[0].isalpha() or len(w) <= 3 for w in words)
    return False


def split_sections(text: str, first_heading: str = "") -> tuple[list[tuple[str, str]], str]:
    """Return ([(heading, body), ...], last_heading). The last heading carries over to the next PDF page."""
    lines = text.split("\n")
    sections, buf, heading = [], [], first_heading
    for i, line in enumerate(lines):
        prev_blank = i == 0 or not lines[i - 1].strip()
        next_has_text = i + 1 < len(lines) and bool(lines[i + 1].strip())
        if looks_like_heading(line, prev_blank, next_has_text):
            if buf:
                sections.append((heading, "\n".join(buf).strip()))
                buf = []
            heading = re.sub(r"^#+\s*", "", line.strip())
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf).strip()))
    return [(h, b) for h, b in sections if b], heading


def pack_sentences(sentences: list[str], max_words: int, overlap_words: int, min_words: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for sentence in sentences:
        n = len(sentence.split())
        if n > max_words:                                   # very long "sentence": hard split
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_words = [], 0
            words = sentence.split()
            step = max(1, max_words - overlap_words)
            chunks.extend(" ".join(words[i:i + max_words]) for i in range(0, len(words), step))
            continue
        if cur and cur_words + n > max_words:
            chunks.append(" ".join(cur))
            carry, carried = [], 0                          # keep trailing sentences as overlap
            for prev in reversed(cur):
                pw = len(prev.split())
                if carried + pw > overlap_words:
                    break
                carry.insert(0, prev)
                carried += pw
            cur, cur_words = carry, carried
        cur.append(sentence)
        cur_words += n
    if cur:
        chunks.append(" ".join(cur))
    if len(chunks) > 1 and len(chunks[-1].split()) < min_words:   # merge a tiny tail
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()
    return chunks


def chunk_section_body(body: str, max_words: int, overlap_words: int, min_words: int) -> list[str]:
    return pack_sentences(split_sentences(body), max_words, overlap_words, min_words)
