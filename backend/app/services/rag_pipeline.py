"""
RAG = Retrieve relevant clauses first, THEN ask Gemini to answer.

This is the core "Phase 2" path when a user asks a question on an already-uploaded contract.
We never paste the full 50-page PDF here — only the reranked top chunks.
"""

import logging

import google.generativeai as genai

from app.config import get_settings
from app.prompts.rag_prompt import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE
from app.schemas.qa import CitationOut
from app.services.retriever import HybridRetriever
from app.utils.language import cross_lingual_instruction, normalize_language

logger = logging.getLogger(__name__)
settings = get_settings()


class RAGPipeline:
    def __init__(self):
        genai.configure(api_key=settings.google_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)
        self.retriever = HybridRetriever()

    def _build_context(self, ranked: list) -> tuple[str, list[CitationOut]]:
        """
        Turn retrieved chunks into numbered [Source N] blocks for the LLM prompt.
        Same numbering drives citations in the API response (page, section, excerpt).
        """
        blocks: list[str] = []
        citations: list[CitationOut] = []
        for i, (chunk, score) in enumerate(ranked, 1):
            header = f"[Source {i}]"
            meta = []
            if chunk.page_number:
                meta.append(f"Page {chunk.page_number}")
            if chunk.section_name:
                meta.append(f"Section: {chunk.section_name}")
            meta_str = " | ".join(meta)
            blocks.append(f"{header} ({meta_str})\n{chunk.chunk_text}")
            citations.append(
                CitationOut(
                    page=chunk.page_number,
                    section=chunk.section_name,
                    clause_number=chunk.metadata.get("clause_number") if chunk.metadata else None,
                    char_offset=chunk.metadata.get("char_start") if chunk.metadata else None,
                    excerpt=chunk.chunk_text[:300],
                )
            )
        return "\n\n---\n\n".join(blocks), citations

    def ask(
        self,
        document_id: int,
        question: str,
        language: str = "en",
        doc_language: str = "en",
    ) -> dict:
        # 1) FAISS hybrid retrieve + BGE rerank (see retriever.py)
        ranked = self.retriever.retrieve(document_id, question)
        if not ranked:
            # nothing in the index — don't let Gemini invent an answer
            return {
                "answer": "Answer not found in the document.",
                "citations": [],
                "confidence": 0.0,
                "sources": [],
            }

        context, citations = self._build_context(ranked)

        # 2) Handle Hindi question on English contract (etc.)
        q_lang = normalize_language(language)
        d_lang = normalize_language(doc_language)
        lang_inst = cross_lingual_instruction(q_lang, d_lang)
        system = RAG_SYSTEM_PROMPT.format(
            language_instruction=f"Respond in {q_lang}. {lang_inst}".strip()
        )
        user = RAG_USER_TEMPLATE.format(context=context, question=question)
        prompt = f"{system}\n\n{user}"

        # 3) Gemini generates — prompt forces grounding + explicit "not found" wording
        response = self.model.generate_content(prompt)
        answer = (response.text or "").strip()

        # belt-and-suspenders: short "not found" style answers get normalized
        if not answer or "not found" in answer.lower() and len(answer) < 80:
            answer = "Answer not found in the document."

        # rough confidence from reranker score (UI shows this on Chat page)
        top_score = float(ranked[0][1]) if ranked else 0.0
        confidence = min(1.0, max(0.0, top_score / 10.0)) if top_score > 1 else min(1.0, top_score)

        return {
            "answer": answer,
            "citations": [c.model_dump() for c in citations],
            "confidence": confidence,
            "sources": [f"Source {i+1}" for i in range(len(ranked))],
        }
