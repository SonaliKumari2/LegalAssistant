"""Build representative document excerpts for classification (not just first N chars)."""

import re
from app.models.document_object import DocumentObject

HEADING_LINE = re.compile(
    r"(?im)^(?:\d+[\.\)]\s*)?(Termination|Liability|Payment|Confidentiality|"
    r"Arbitration|Indemnification|Employment|Lease|Rent|NDA|Services|Definitions)\b.*$"
)


def extract_representative_text(doc: DocumentObject, max_chars: int = 6000) -> str:
    """
    Prefer titles/headings + stratified samples (top, middle, bottom).
    More robust than using only the first 5000 characters.
    """
    parts: list[str] = []

    if doc.title:
        parts.append(f"TITLE: {doc.title}")

    heading_lines: list[str] = []
    for line in doc.text.splitlines():
        if HEADING_LINE.match(line.strip()) and len(line.strip()) < 200:
            heading_lines.append(line.strip())
    if heading_lines:
        parts.append("HEADINGS:\n" + "\n".join(heading_lines[:25]))

    text = doc.text
    n = len(text)
    if n == 0:
        return ""

    slices = [
        ("BEGIN", text[: max_chars // 3]),
        ("MIDDLE", text[max(0, n // 2 - max_chars // 6) : max(0, n // 2 - max_chars // 6) + max_chars // 3]),
        ("END", text[max(0, n - max_chars // 3) :]),
    ]
    for label, segment in slices:
        if segment.strip():
            parts.append(f"{label}:\n{segment.strip()}")

    combined = "\n\n---\n\n".join(parts)
    return combined[:max_chars]
