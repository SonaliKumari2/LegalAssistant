"""
PDF ingestion — we use PyMuPDF (fitz) instead of PyPDF2 for better text + layout.

Outputs a DocumentObject:
  - full text (for chunking)
  - per-page text (for citations — "see page 12")
  - detected section headings on each page (helps structure chunker)
PII is redacted here before text is stored anywhere else.
"""

import logging
import re
from pathlib import Path
from typing import BinaryIO

import fitz

from app.models.document_object import DocumentObject, PageContent
from app.utils.pii_redaction import redact_pii

logger = logging.getLogger(__name__)

# common clause titles — structure_chunker has a bigger list; this is per-page hints
HEADING_PATTERN = re.compile(
    r"^(?:\d+\.?\s+)?(Termination|Liability|Payment Terms|Confidentiality|Arbitration|"
    r"Indemnification|Governing Law|Definitions|Scope of Work)\b",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_sections(page_text: str) -> list[str]:
    return [m.group(1) for m in HEADING_PATTERN.finditer(page_text)]


def parse_pdf(source: str | Path | BinaryIO) -> DocumentObject:
    if isinstance(source, (str, Path)):
        doc = fitz.open(str(source))
        metadata = doc.metadata or {}
    else:
        data = source.read()
        doc = fitz.open(stream=data, filetype="pdf")
        metadata = doc.metadata or {}

    pages: list[PageContent] = []
    full_parts: list[str] = []

    for i, page in enumerate(doc):
        page_num = i + 1
        text = page.get_text("text") or ""
        text = redact_pii(text)
        sections = _extract_sections(text)
        pages.append(PageContent(page_number=page_num, text=text, sections=sections))
        full_parts.append(text)

    doc.close()
    full_text = "\n\n".join(full_parts)
    title = metadata.get("title") or (full_text[:120].split("\n")[0][:80] if full_text else "Untitled")

    return DocumentObject(
        text=full_text,
        pages=pages,
        metadata={
            "author": metadata.get("author"),
            "subject": metadata.get("subject"),
            "creator": metadata.get("creator"),
            "format": "pdf",
        },
        title=title,
    )
