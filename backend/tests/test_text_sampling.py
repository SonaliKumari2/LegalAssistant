from app.models.document_object import DocumentObject, PageContent
from app.utils.text_sampling import extract_representative_text


def test_representative_includes_headings_and_sections():
    text = "EMPLOYMENT AGREEMENT\n\n1. Termination\nEmployee may be terminated...\n\n" + ("body " * 500)
    doc = DocumentObject(
        text=text,
        pages=[PageContent(1, text, ["Termination"])],
        metadata={},
        title="Employment Agreement",
    )
    rep = extract_representative_text(doc, max_chars=8000)
    assert "TITLE:" in rep or "BEGIN" in rep
    assert "Termination" in rep or "EMPLOYMENT" in rep
