"""
Figure out what kind of contract this is before we chunk or extract risks.

Why classify?
  - employment vs NDA vs lease use different risk checklists (JSON frameworks)
  - we can add type-specific legal headings when structure-chunking

Confidence gap trick (interview favorite):
  don't trust only the top score — compare top TWO scores.
  If they're too close (gap < 0.05), ask the human to pick the type.
"""

import json
import logging
import re
from typing import Callable

import google.generativeai as genai
import numpy as np

from app.config import get_settings
from app.models.document_object import DocumentObject
from app.utils.text_sampling import extract_representative_text

logger = logging.getLogger(__name__)
settings = get_settings()

DOCUMENT_TYPES = [
    "Employment Contract",
    "Lease Agreement",
    "Rental Agreement",
    "NDA",
    "Vendor Agreement",
    "Service Agreement",
    "General Legal Document",
]

CLASSIFY_PROMPT = """You are a legal document classifier.

Classify this contract into categories with confidence scores (0-1, approximately summing to 1).
Use headings, obligations, and party language — not surface keywords alone.

Categories: {categories}

Representative document excerpt:
{text}

{embedding_hint}

Return JSON only:
{{"scores": {{"Employment Contract": 0.0, ...}}, "detected_language": "en|hi|hinglish|other"}}
"""


def _heuristic_scores(text: str) -> dict[str, float]:
    """Fallback if Gemini returns garbage — keyword hit counts, normalized."""
    text_l = text.lower()
    keywords = {
        "Employment Contract": ["employee", "employer", "salary", "probation", "non-compete"],
        "Lease Agreement": ["lessor", "lessee", "premises", "lease term"],
        "Rental Agreement": ["tenant", "landlord", "rent", "security deposit"],
        "NDA": ["confidential information", "non-disclosure", "nda"],
        "Vendor Agreement": ["vendor", "supplier", "purchase order"],
        "Service Agreement": ["services", "service level", "sow", "statement of work"],
    }
    scores = {t: 0.05 for t in DOCUMENT_TYPES}
    for dtype, kws in keywords.items():
        hits = sum(1 for k in kws if k in text_l)
        scores[dtype] = 0.1 + hits * 0.15
    total = sum(scores.values()) or 1
    return {k: v / total for k, v in scores.items()}


def _sample_segments(text: str, count: int = 8, segment_len: int = 500) -> list[str]:
    """
    Grab snippets from start, middle, end of the doc — better than only first 5000 chars
    when the title page doesn't say what kind of contract it is.
    """
    if len(text) <= segment_len:
        return [text]
    positions = [0, len(text) // 4, len(text) // 2, 3 * len(text) // 4, max(0, len(text) - segment_len)]
    segments: list[str] = []
    for start in positions:
        seg = text[start : start + segment_len].strip()
        if seg and seg not in segments:
            segments.append(seg)
    while len(segments) < count and len(text) > segment_len:
        step = max(segment_len, len(text) // count)
        for i in range(0, len(text), step):
            seg = text[i : i + segment_len].strip()
            if seg and len(segments) < count:
                segments.append(seg)
    return segments[:count]


def _average_embedding(embed_fn: Callable[[list[str]], list[list[float]]], text: str) -> list[float] | None:
    """
    Optional: average a few segment embeddings into one 'document vector'.
    We don't classify FROM this vector directly — we pass a hint to Gemini.
    Cheaper than sending every chunk to the LLM for classification.
    """
    segments = _sample_segments(text)
    if not segments:
        return None
    try:
        vectors = embed_fn(segments)
        matrix = np.array(vectors, dtype=np.float32)
        return matrix.mean(axis=0).tolist()
    except Exception as exc:
        logger.warning("Document embedding average failed: %s", exc)
        return None


class DocumentClassifier:
    def __init__(self, confidence_gap: float | None = None):
        self.confidence_gap = confidence_gap or settings.classification_confidence_gap
        genai.configure(api_key=settings.google_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)

    def classify(self, text: str, embed_fn: Callable[[list[str]], list[list[float]]] | None = None) -> dict:
        excerpt = extract_representative_text_from_plain(text)
        return self._classify_excerpt(excerpt, text, embed_fn)

    def classify_document(
        self,
        doc: DocumentObject,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> dict:
        excerpt = extract_representative_text(doc)
        return self._classify_excerpt(excerpt, doc.text, embed_fn)

    def _classify_excerpt(
        self,
        excerpt: str,
        full_text: str,
        embed_fn: Callable[[list[str]], list[list[float]]] | None,
    ) -> dict:
        embedding_hint = ""
        if embed_fn:
            avg = _average_embedding(embed_fn, full_text)
            if avg:
                embedding_hint = (
                    "A document-level embedding was computed (mean of stratified segment embeddings). "
                    f"Vector dimension: {len(avg)}. Use excerpt + legal cues for classification."
                )

        try:
            prompt = CLASSIFY_PROMPT.format(
                categories=", ".join(DOCUMENT_TYPES),
                text=excerpt,
                embedding_hint=embedding_hint,
            )
            response = self.model.generate_content(prompt)
            raw = response.text or ""
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                scores = data.get("scores", {})
                lang = data.get("detected_language", "en")
            else:
                raise ValueError("No JSON in response")
        except Exception as exc:
            logger.warning("LLM classification failed, using heuristics: %s", exc)
            scores = _heuristic_scores(full_text)
            lang = "en"

        return self._apply_confidence_gap(scores, lang)

    def _apply_confidence_gap(self, scores: dict, lang: str) -> dict:
        sorted_types = sorted(scores.items(), key=lambda x: float(x[1]), reverse=True)
        top_type, top_score = sorted_types[0]
        second_score = float(sorted_types[1][1]) if len(sorted_types) > 1 else 0.0
        top_score = float(top_score)
        gap = top_score - second_score
        manual = gap < self.confidence_gap

        return {
            "document_type": None if manual else top_type,
            "manual_selection_required": manual,
            "classification_scores": {k: float(v) for k, v in scores.items()},
            "confidence_gap": gap,
            "detected_language": lang,
            "top_candidates": sorted_types[:3],
        }


def extract_representative_text_from_plain(text: str) -> str:
    lines = text.splitlines()
    fake_pages = [type("P", (), {"page_number": 1, "text": text, "sections": []})()]
    doc = DocumentObject(text=text, pages=fake_pages, metadata={}, title=lines[0][:80] if lines else None)
    return extract_representative_text(doc)
