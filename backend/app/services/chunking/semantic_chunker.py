import logging
import uuid

import numpy as np

from app.models.document_object import DocumentObject
from app.services.chunking.structure_chunker import TextChunk, _approx_tokens, _page_for_offset

logger = logging.getLogger(__name__)


def semantic_chunk(
    doc: DocumentObject,
    target_tokens: tuple[int, int],
    embed_fn=None,
) -> list[TextChunk]:
    min_tok, max_tok = target_tokens
    sentences = _split_sentences(doc.text)
    if len(sentences) < 3:
        return _fallback_fixed(doc, min_tok, max_tok)

    if embed_fn is None:
        return _fallback_fixed(doc, min_tok, max_tok)

    try:
        embeddings = embed_fn(sentences)
        embs = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        embs = embs / norms
        similarities = np.sum(embs[:-1] * embs[1:], axis=1)
        threshold = float(np.percentile(similarities, 25))
        breakpoints = [0] + [i + 1 for i, s in enumerate(similarities) if s < threshold] + [len(sentences)]
    except Exception as exc:
        logger.warning("Semantic chunking failed: %s", exc)
        return _fallback_fixed(doc, min_tok, max_tok)

    chunks: list[TextChunk] = []
    offset = 0
    for i in range(len(breakpoints) - 1):
        seg_sentences = sentences[breakpoints[i] : breakpoints[i + 1]]
        segment = " ".join(seg_sentences).strip()
        if not segment:
            continue
        start = doc.text.find(segment, offset)
        if start < 0:
            start = offset
        offset = start + len(segment)
        page = _page_for_offset(doc, start)
        for sub in _window_split(segment, min_tok, max_tok):
            chunks.append(
                TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=sub,
                    page_number=page,
                    section_name=None,
                    char_start=start,
                    char_end=start + len(sub),
                    chunk_size="small" if max_tok <= 400 else "large",
                )
            )
    return chunks


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _window_split(text: str, min_tok: int, max_tok: int) -> list[str]:
    words = text.split()
    if _approx_tokens(text) <= max_tok:
        return [text]
    parts: list[str] = []
    buf: list[str] = []
    for w in words:
        buf.append(w)
        if _approx_tokens(" ".join(buf)) >= max_tok:
            parts.append(" ".join(buf))
            buf = []
    if buf:
        parts.append(" ".join(buf))
    return parts


def _fallback_fixed(doc: DocumentObject, min_tok: int, max_tok: int) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    words = doc.text.split()
    step = max_tok * 4 // 5
    offset = 0
    for i in range(0, len(words), step):
        segment = " ".join(words[i : i + step])
        if not segment:
            continue
        start = doc.text.find(segment, offset)
        chunks.append(
            TextChunk(
                chunk_id=str(uuid.uuid4()),
                text=segment,
                page_number=_page_for_offset(doc, max(start, 0)),
                section_name=None,
                char_start=max(start, 0),
                char_end=max(start, 0) + len(segment),
                chunk_size="small" if max_tok <= 400 else "large",
            )
        )
        offset = start + len(segment)
    return chunks
