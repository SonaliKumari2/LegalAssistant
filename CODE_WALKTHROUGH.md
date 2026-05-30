# Code walkthrough — what to open when explaining in an interview

Read the **comments inside each file** — they're written so you can explain aloud while scrolling.

| If they ask about… | Open this file |
|--------------------|----------------|
| Whole upload flow | `backend/app/services/document_ingestion.py` |
| PDF parsing | `backend/app/services/parsers/pdf_parser.py` |
| What is RAG / Q&A | `backend/app/services/rag_pipeline.py` |
| Top-5 small + large + rerank | `backend/app/services/retriever.py` |
| BGE cross-encoder | `backend/app/services/reranker.py` |
| FAISS / cosine | `backend/app/services/vector_store.py` |
| Google embeddings | `backend/app/services/embedding_service.py` |
| Hybrid / structure chunking | `backend/app/services/chunking/hybrid_chunker.py` |
| Classification + confidence gap | `backend/app/services/document_classifier.py` |
| Risk frameworks | `backend/app/services/risk_analyzer.py` + `risk_frameworks/*.json` |
| Strict LLM prompt | `backend/app/prompts/rag_prompt.py` |
| RAGAS metrics | `backend/app/evaluation/ragas_evaluator.py` |
| Risk precision/recall | `backend/app/evaluation/risk_metrics.py` |
| API wiring | `backend/app/main.py` |
| Old prototype (contrast only) | `legalDocumentAssistant/app.py` |

**Live demo endpoints:** `GET /api/features/` · `GET /api/documents/{id}/pipeline`

---

## Frontend (UI → API)

| If they ask about… | Open this file |
|--------------------|----------------|
| HTTP / JWT from browser | `frontend/src/services/api.js` |
| Login state | `frontend/src/contexts/AuthContext.jsx` |
| Routes / protected pages | `frontend/src/App.jsx` |
| Upload flow | `frontend/src/pages/UploadPage.jsx` |
| RAG chat UI | `frontend/src/pages/ChatPage.jsx` |
| Pipeline checkmarks demo | `frontend/src/pages/DocumentViewer.jsx` |
| Manual doc type picker | `frontend/src/pages/Dashboard.jsx` |
| Risks UI | `frontend/src/pages/RiskPage.jsx` |
| Summary UI | `frontend/src/pages/SummaryPage.jsx` |
| RAGAS eval UI | `frontend/src/pages/EvaluationPage.jsx` |

## API layer (thin wrappers over services)

| If they ask about… | Open this file |
|--------------------|----------------|
| POST /qa/ask | `backend/app/api/qa.py` |
| Upload endpoint | `backend/app/api/documents.py` |
| Login / register | `backend/app/api/auth.py` |
| Risk endpoints | `backend/app/api/risks.py` |
| Summary endpoint | `backend/app/api/summaries.py` |
| JWT guard | `backend/app/api/deps.py` |
