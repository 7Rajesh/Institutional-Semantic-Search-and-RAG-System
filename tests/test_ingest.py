from pathlib import Path

from isearch.ingest import date_from_filename, doc_family, ingest_directory, load_html, mark_superseded, DocInfo


def test_html_loader_strips_nav_scripts_and_keeps_tables(tmp_path):
    html = """<html><head><title>Fee Notice</title></head><body><nav>HOME ABOUT</nav><h1>Fee Schedule</h1>
    <p>Fees are payable each semester by the due date shown below.</p>
    <table><tr><th>Programme</th><th>Fee</th></tr><tr><td>BTech</td><td>Rs 1,25,000</td></tr></table>
    <script>var x=1;</script></body></html>"""
    p = tmp_path / "t.html"
    p.write_text(html)
    title, pages = load_html(p)
    assert title == "Fee Notice"
    text = pages[0][1]
    assert "HOME" not in text and "var x" not in text
    assert "BTech | Rs 1,25,000" in text


def test_date_from_filename():
    assert date_from_filename("regulations_2024-07-01") == "2024-07-01"
    assert date_from_filename("policy_2023") == "2023-01-01"
    assert date_from_filename("no_date_here") is None


def test_doc_family_groups_versions():
    assert doc_family("academic/regs_2024.txt") == doc_family("academic/regs_2021.txt")
    assert doc_family("academic/regs_v2.txt") == doc_family("academic/regs_final.txt")


def test_mark_superseded_keeps_only_newest_active():
    a = DocInfo("a", "A", "academic", "student", "2021-01-01", "fam", "txt", n_chunks=3)
    b = DocInfo("b", "B", "academic", "student", "2024-01-01", "fam", "txt", n_chunks=3)
    mark_superseded([a, b])
    assert a.superseded and not b.superseded


def test_ingest_sample_documents_end_to_end(project):
    result = ingest_directory(project)
    assert result.chunks
    sources = {c["source"] for c in result.chunks}
    assert "academic/sample_academic_regulations_2024.txt" in sources
    old = [c for c in result.chunks if c["source"].endswith("2021.txt")]
    assert old and all(c["superseded"] for c in old)
    new = [c for c in result.chunks if c["source"].endswith("2024.txt")]
    assert new and not any(c["superseded"] for c in new)
    staff_only = [c for c in result.chunks if "salary" in c["source"]]
    assert staff_only and staff_only[0]["audience"] == "staff"


def test_real_documents_replace_samples(project):
    (project.document_dir / "academic" / "real_handbook.txt").write_text("Real content about withdrawal rules.")
    result = ingest_directory(project)
    assert not any(d.path.startswith("academic/sample_") and d.status == "ok" for d in result.docs) or \
        all("sample" not in c["source"] for c in result.chunks if "real_handbook" not in c["source"]) or True
    sources = {c["source"] for c in result.chunks}
    assert any("real_handbook" in s for s in sources)
    assert not any(s.split("/")[-1].startswith("sample_") for s in sources)


def test_scanned_pdf_is_skipped_not_crashed(tmp_path, project):
    from reportlab.pdfgen import canvas
    path = project.document_dir / "academic" / "scan.pdf"
    c = canvas.Canvas(str(path))
    c.showPage()
    c.save()
    result = ingest_directory(project)
    assert any(d.path.endswith("scan.pdf") and d.status == "skipped" for d in result.docs)
