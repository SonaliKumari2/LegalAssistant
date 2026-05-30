"""
Split contracts at legal section headings (Termination, Liability, …).

Legal docs are usually structured — this keeps whole clauses together.
If we can't find enough headings, hybrid_chunker falls back to semantic_chunker.

document_type adds extra headings (e.g. "Rent" for leases) via document_type_patterns.py
"""

import re
import uuid
from dataclasses import dataclass

from app.models.document_object import DocumentObject
from app.services.chunking.document_type_patterns import TYPE_EXTRA_HEADINGS

BASE_HEADINGS = (
    r"Termination|Liability|Payment\s*Terms?|Confidentiality|"
    r"Arbitration|Indemnification|Governing\s*Law|Definitions|Scope|Warranties|"
    r"Intellectual\s*Property|Non[- ]?compete|Renewal|Notice"
)


def _heading_regex(document_type: str | None) -> re.Pattern:
    extra = TYPE_EXTRA_HEADINGS.get(document_type or "", [])
    if extra:
        extra_part = "|".join(re.escape(t) for t in extra)
        pattern = rf"(?im)^(?:\d+[\.\)]\s*)?({BASE_HEADINGS}|{extra_part})\b.*$"
    else:
        pattern = rf"(?im)^(?:\d+[\.\)]\s*)?({BASE_HEADINGS})\b.*$"
    return re.compile(pattern)


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    page_number: int | None
    section_name: str | None
    char_start: int
    char_end: int
    chunk_size: str  # small | large


def structure_chunk(
    doc: DocumentObject,
    target_tokens: tuple[int, int],
    document_type: str | None = None,
) -> list[TextChunk]:
    min_tok, max_tok = target_tokens
    full = doc.text
    heading_re = _heading_regex(document_type)
    splits = list(heading_re.finditer(full))
    # need at least a couple of headings — otherwise caller uses semantic chunking
    if len(splits) < 2:
        return []

    chunks: list[TextChunk] = []
    boundaries = [0] + [m.start() for m in splits[1:]] + [len(full)]
    section_names = ["Preamble"] + [m.group(1) for m in splits[1:]]

    for idx in range(len(boundaries) - 1):
        start, end = boundaries[idx], boundaries[idx + 1]
        section = section_names[idx] if idx < len(section_names) else None
        segment = full[start:end].strip()
        if not segment:
            continue
        page = _page_for_offset(doc, start)
        sub_chunks = _split_by_token_window(segment, min_tok, max_tok)
        for sub in sub_chunks:
            chunks.append(
                TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=sub,
                    page_number=page,
                    section_name=section,
                    char_start=start,
                    char_end=start + len(sub),
                    chunk_size="small" if max_tok <= 400 else "large",
                )
            )
    return chunks


def _page_for_offset(doc: DocumentObject, offset: int) -> int | None:
    cursor = 0
    for page in doc.pages:
        cursor += len(page.text) + 2
        if offset < cursor:
            return page.page_number
    return doc.pages[-1].page_number if doc.pages else None


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _split_by_token_window(text: str, min_tok: int, max_tok: int) -> list[str]:
    words = text.split()
    if _approx_tokens(text) <= max_tok:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    for w in words:
        current.append(w)
        if _approx_tokens(" ".join(current)) >= max_tok:
            parts.append(" ".join(current))
            current = []
    if current:
        parts.append(" ".join(current))
    return parts
