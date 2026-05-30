# Feature Verification — Interview Checklist

Use this during demos to prove **each talking point is implemented in code**.

**Live API:** `GET http://localhost:8000/api/features/`  
**Per document:** `GET http://localhost:8000/api/documents/{id}/pipeline`

---

## Feature → Code Map

| # | Interview feature | Implemented | Proof (file / endpoint) |
|---|-------------------|-------------|-------------------------|
| 1 | **RAG definition** | Yes | Retrieve from FAISS → rerank → Gemini (`rag_pipeline.py`) |
| 2 | **PDF/DOCX ingestion** | Yes | `parsers/pdf_parser.py`, `parsers/docx_parser.py` |
| 3 | **Why chunking (not full doc)** | Yes | Dual-size chunks; metadata in `pipeline.chunking` |
| 4 | **Fixed chunking rejected** | Yes | Not used; hybrid structure/semantic only |
| 5 | **Document classification** | Yes | `document_classifier.py` + 7 types |
| 6 | **Confidence gap → manual** | Yes | `classification_confidence_gap` in config; UI on Dashboard |
| 7 | **Representative text (not only first 5k)** | Yes | `utils/text_sampling.py` |
| 8 | **Mean embedding for classification** | Yes | `classify_document(..., embed_fn=...)` |
| 9 | **Structure chunking (preferred)** | Yes | `structure_chunker.py` + type-specific headings |
| 10 | **Semantic fallback** | Yes | `semantic_chunker.py` when no headings |
| 11 | **Small 200–400 / Large 1000–1500** | Yes | `hybrid_chunker.py` `SMALL_RANGE`, `LARGE_RANGE` |
| 12 | **Hybrid retrieval 5+5** | Yes | `retriever.py` + config `retrieval_small_k/large_k` |
| 13 | **Google embeddings** | Yes | `embedding_service.py` |
| 14 | **FAISS + cosine** | Yes | `vector_store.py` L2-normalize + `IndexFlatIP` |
| 15 | **BGE cross-encoder rerank** | Yes | `reranker.py` (default `bge-reranker-large`) |
| 16 | **Answer not in document** | Yes | Exact string in `prompts/rag_prompt.py` |
| 17 | **Citations (page → section → offset)** | Yes | `rag_pipeline.py` → `Citation` model |
| 18 | **Summarization (7 sections)** | Yes | `summarizer.py` + Summary API |
| 19 | **Risk: framework + open-ended** | Yes | `risk_analyzer.py` + `risk_frameworks/*.json` |
| 20 | **Multilingual EN/HI/Hinglish** | Yes | `utils/language.py` |
| 21 | **RAGAS evaluation** | Yes | `evaluation/ragas_evaluator.py` + Evaluation page |
| 22 | **Risk precision/recall** | Yes | `evaluation/risk_metrics.py` |
| 23 | **JWT auth** | Yes | `api/auth.py` |
| 24 | **PII redaction** | Yes | `utils/pii_redaction.py` at parse time |
| 25 | **TTL / cleanup** | Yes | `document_ttl_hours` + `cleanup_expired.py` |

---

## Demo Script (5 minutes)

1. **Register / Login** → http://localhost:5173  
2. **Upload** a PDF employment contract → see classification scores  
3. If amber **“Manual type required”** → pick type on Dashboard → saves via API  
4. **Chat** → ask *“What is the notice period?”* → answer + page citations  
5. **Risks** → framework + open-ended risks with severity  
6. **Summary** → switch English/Hindi  
7. **Evaluation** → run RAGAS on 2–3 questions  
8. **Swagger** → `GET /api/features/` and `GET /api/documents/1/pipeline`  

---

## What NOT to demo for full pipeline

| App | Path | Has full pipeline? |
|-----|------|-------------------|
| **Kanooni Sahayak** | `kanooni-sahayak/` | **YES — use this in interviews** |
| Streamlit prototype | `legalDocumentAssistant/app.py` | **NO** — basic fixed chunking only |

---

## Reported benchmark numbers (your validation set)

| Metric | Value | How measured |
|--------|-------|----------------|
| Risk precision | ~88% | Manual labels on ~20 contracts |
| Risk recall | ~84% | Same set; recall prioritized |
| RAGAS faithfulness | ~92% | `POST /api/evaluation/run` |
| RAGAS answer relevance | ~90% | Same |
| Retrieval validation | ~30 docs | Manual Q → expected clause check |

*Disclose: not reviewed by a licensed attorney; frameworks from common legal checklists.*

---

## Quick verbal answers

- **Why FAISS?** Local, free, private — no vector SaaS API.  
- **Why BGE not MiniLM?** Legal needs relevance over speed.  
- **Why Google embeddings?** Multilingual; tradeoff vs Legal-BERT.  
- **Chunk size validation?** RAGAS grid on 20–30 questions per doc type.
