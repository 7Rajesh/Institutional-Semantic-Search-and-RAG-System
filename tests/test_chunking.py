from isearch.chunking import chunk_section_body, looks_like_heading, split_sections
from isearch.text import split_sentences


def test_split_sentences_respects_abbreviations():
    out = split_sentences("Dr. Anita Sharma teaches here. Rs 500 is due.")
    assert out == ["Dr. Anita Sharma teaches here.", "Rs 500 is due."]


def test_heading_detection():
    assert looks_like_heading("1. Course Withdrawal", True, True)
    assert looks_like_heading("ATTENDANCE REQUIREMENT", True, True)
    assert not looks_like_heading("This is a normal sentence that ends with a period.", True, True)


def test_split_sections_carries_heading_across_calls():
    text = "1. Course Withdrawal\n\nStudents may withdraw until week 8."
    sections, last = split_sections(text)
    assert sections == [("1. Course Withdrawal", "Students may withdraw until week 8.")]
    more, last2 = split_sections("More detail on withdrawal follows here.", first_heading=last)
    assert more == [("1. Course Withdrawal", "More detail on withdrawal follows here.")]


def test_chunk_respects_max_words_and_overlaps():
    text = " ".join(f"Sentence number {i} says that students must comply with rule {i}." for i in range(60))
    chunks = chunk_section_body(text, max_words=180, overlap_words=30, min_words=15)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 180 for c in chunks)
    # the last sentence of chunk 1 should reappear at the start of chunk 2 (overlap)
    assert chunks[0].split(".")[-2].strip() in chunks[1]


def test_chunk_handles_a_single_giant_sentence():
    giant = " ".join(["word"] * 500)
    chunks = chunk_section_body(giant, max_words=180, overlap_words=30, min_words=15)
    assert all(len(c.split()) <= 180 for c in chunks)
    assert len(chunks) >= 3


def test_tiny_tail_chunk_gets_merged():
    text = " ".join(f"Sentence {i} is here to fill space nicely." for i in range(40)) + " Short end."
    chunks = chunk_section_body(text, max_words=180, overlap_words=30, min_words=15)
    assert len(chunks[-1].split()) >= 15 or len(chunks) == 1
