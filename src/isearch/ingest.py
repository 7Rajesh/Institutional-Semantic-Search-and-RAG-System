"""Document loading, metadata (category, audience, effective date, versions) and chunking."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from .chunking import chunk_section_body, split_sections
from .config import Settings
from .text import clean_text

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".html", ".htm"}
AUDIENCES = ("public", "student", "staff")

CATEGORY_KEYWORDS = {
    "academic":   ["academic", "course", "curriculum", "regulation", "exam", "syllabus", "registration"],
    "admissions": ["admission", "brochure", "eligibility", "fee", "scholarship"],
    "hostel":     ["hostel", "mess", "residence"],
    "research":   ["research", "phd", "thesis", "laboratory"],
    "faculty":    ["faculty", "professor", "staff", "profile"],
    "policy":     ["policy", "notice", "circular", "conduct", "rules"],
}


@dataclass
class DocInfo:
    path: str
    title: str
    category: str
    audience: str
    effective: str | None
    family: str
    source_type: str
    n_chunks: int = 0
    superseded: bool = False
    status: str = "ok"          # ok | skipped
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IngestResult:
    chunks: list[dict]
    docs: list[DocInfo]


# ---------------------------------------------------------------- loaders
def strip_repeated_lines(pages: list[str], min_pages: int = 3, share: float = 0.6) -> list[str]:
    """Drop running headers/footers: short lines that appear on most pages."""
    if len(pages) < min_pages:
        return pages
    norm = lambda line: re.sub(r"\d+", "#", line.strip().lower())
    counts: dict[str, int] = {}
    for text in pages:
        for line in {norm(l) for l in text.split("\n") if l.strip()}:
            counts[line] = counts.get(line, 0) + 1
    repeated = {l for l, c in counts.items() if c >= share * len(pages) and len(l) < 100}
    return ["\n".join(l for l in text.split("\n") if norm(l) not in repeated) for text in pages]


def load_pdf(path: Path):
    reader = PdfReader(str(path))
    pages = strip_repeated_lines([clean_text(p.extract_text() or "") for p in reader.pages])
    title = (reader.metadata.title if reader.metadata else None) or path.stem
    return title, list(enumerate(pages, start=1))


def load_html(path: Path):
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else path.stem
    if soup.head:
        soup.head.decompose()                                 # keep <title> text out of the body
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside"]):
        tag.decompose()
    for h in soup.find_all(re.compile(r"^h[1-6]$")):          # headings -> markdown headings
        h.replace_with("\n\n# " + h.get_text(" ", strip=True) + "\n")
    for row in soup.find_all("tr"):                           # table rows -> "a | b | c"
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
        row.replace_with("\n" + " | ".join(cells) + "\n")
    return title, [(None, clean_text(soup.get_text("\n")))]


_FRONT = re.compile(r"^(title|effective|audience|category|version)\s*:\s*(.+?)\s*$", re.I)


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Optional `Key: value` lines at the top of a .txt/.md file, followed by a blank line."""
    lines = text.split("\n")
    meta, i = {}, 0
    while i < len(lines) and _FRONT.match(lines[i].strip()):
        k, v = _FRONT.match(lines[i].strip()).groups()
        meta[k.lower()] = v
        i += 1
    if not meta:
        return {}, text
    return meta, "\n".join(lines[i:]).lstrip("\n")


def load_text(path: Path):
    raw = path.read_text(encoding="utf-8", errors="ignore")
    meta, body = parse_front_matter(raw)
    title = meta.get("title") or path.stem.replace("_", " ").title()
    return title, [(None, clean_text(body))], meta


def load_document(path: Path):
    """Return (title, pages, embedded_metadata)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        title, pages = load_pdf(path)
        return title, pages, {}
    if suffix in {".html", ".htm"}:
        title, pages = load_html(path)
        return title, pages, {}
    return load_text(path)


# ---------------------------------------------------------------- metadata
def infer_category(path: Path, directory: Path) -> str:
    parts = path.relative_to(directory).parts
    if len(parts) > 1:
        return parts[0].lower()
    name = path.stem.lower()
    hits = {cat: sum(k in name for k in kws) for cat, kws in CATEGORY_KEYWORDS.items()}
    best = max(hits, key=hits.get)
    return best if hits[best] > 0 else "general"


def _valid_date(text: str | None) -> str | None:
    if not text:
        return None
    try:
        return date.fromisoformat(text.strip()[:10]).isoformat()
    except ValueError:
        return None


def date_from_filename(stem: str) -> str | None:
    m = re.search(r"(?<!\d)(\d{4})[-_](\d{2})[-_](\d{2})(?!\d)", stem)
    if m:
        return _valid_date("-".join(m.groups()))
    m = re.search(r"(?<!\d)(20\d{2})(?!\d)", stem)
    return f"{m.group(1)}-01-01" if m else None


def read_sidecar(path: Path) -> dict:
    side = path.with_name(path.name + ".meta.json")
    if side.exists():
        try:
            return json.loads(side.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


_VERSION_TAIL = re.compile(r"[_\-\s]*(v\d+(\.\d+)*|\d{4}([_\-]\d{2}){0,2}|final|latest|draft)$")


def doc_family(rel_path: str) -> str:
    """Files that differ only by a date/version suffix belong to one family (regulations_2023, regulations_2024)."""
    p = Path(rel_path)
    stem, prev = p.stem.lower(), None
    while prev != stem:
        prev, stem = stem, _VERSION_TAIL.sub("", stem)
    return f"{p.parent.as_posix()}/{stem}"


def mark_superseded(docs: list[DocInfo]) -> None:
    """Within a family, every document older than the newest one is marked superseded."""
    families: dict[str, list[DocInfo]] = {}
    for d in docs:
        if d.status == "ok":
            families.setdefault(d.family, []).append(d)
    for members in families.values():
        if len(members) < 2:
            continue
        newest = max(m.effective or "" for m in members)
        for m in members:
            if (m.effective or "") < newest:
                m.superseded = True


# ---------------------------------------------------------------- ingestion
def ingest_directory(settings: Settings) -> IngestResult:
    directory = settings.document_dir
    files = [p for p in sorted(directory.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    real_files = [p for p in files if not p.name.startswith("sample_")]
    if real_files:
        files = real_files                                  # real documents replace the samples

    chunks: list[dict] = []
    docs: list[DocInfo] = []
    seen: set[str] = set()

    for path in files:
        rel = path.relative_to(directory).as_posix()
        try:
            title, pages, embedded = load_document(path)
        except Exception as exc:
            docs.append(DocInfo(rel, path.stem, "general", "public", None, doc_family(rel), path.suffix.lstrip("."),
                                status="skipped", note=f"could not read: {exc}"))
            continue

        meta = {**embedded, **read_sidecar(path)}
        audience = str(meta.get("audience", "public")).lower()
        if audience not in AUDIENCES:
            audience = "public"
        info = DocInfo(
            path=rel,
            title=str(meta.get("title") or title),
            category=str(meta.get("category") or infer_category(path, directory)).lower(),
            audience=audience,
            effective=_valid_date(meta.get("effective")) or date_from_filename(path.stem),
            family=doc_family(rel),
            source_type=path.suffix.lower().lstrip("."),
        )

        heading = ""
        for page_no, text in pages:
            sections, heading = split_sections(text, heading)
            for section, body in sections:
                for piece in chunk_section_body(body, settings.chunk_words, settings.chunk_overlap_words,
                                                settings.min_chunk_words):
                    key = hashlib.md5((rel + re.sub(r"\W+", " ", piece.lower())).encode()).hexdigest()
                    if key in seen:
                        continue
                    seen.add(key)
                    chunks.append({
                        "chunk_id": f"{rel}:{info.n_chunks}",
                        "source": rel,
                        "title": info.title,
                        "category": info.category,
                        "audience": info.audience,
                        "effective": info.effective,
                        "family": info.family,
                        "superseded": False,                # filled in below
                        "source_type": info.source_type,
                        "page": page_no,
                        "section": section,
                        "text": piece,
                        "context_text": f"{info.title} | {section}\n{piece}" if section else f"{info.title}\n{piece}",
                    })
                    info.n_chunks += 1
        if info.n_chunks == 0:
            info.status, info.note = "skipped", "no extractable text (scanned PDF? it needs OCR first)"
        docs.append(info)

    mark_superseded(docs)
    superseded = {d.path for d in docs if d.superseded}
    for c in chunks:
        c["superseded"] = c["source"] in superseded
    return IngestResult(chunks, docs)
