# Kanooni Sahayak — Read-Aloud Code Script (Line by Line)

Use this file to **practice explaining the project aloud** in an interview or demo.  
Each section follows the real source files. Line numbers match the current codebase.

**Tip:** You do not need to read every line in an interview. Pick one path:
- **Upload path:** `run.py` → `documents.py` (upload) → `document_ingestion.py`
- **Question path:** `ChatPage.jsx` → `qa.py` → `rag_pipeline.py` → `retriever.py`

---

## Table of contents

1. [How the project starts](#1-backend-runpy--mainpy)
2. [Configuration & database](#2-configpy--databasepy)
3. [Upload API](#3-apidocumentspy-upload-endpoint)
4. [Ingestion pipeline (core)](#4-document_ingestionpy)
5. [Hybrid chunking](#5-hybrid_chunkerpy)
6. [Hybrid retrieval](#6-retrieverpy)
7. [RAG pipeline](#7-rag_pipelinepy)
8. [Q&A API](#8-apiqapy)
9. [Frontend chat](#9-frontend-chatpagejsx)
10. [Frontend API client](#10-frontend-apijs)
11. [Old Streamlit prototype (contrast)](#11-legaldocumentassistant-apppy)

---

## 1. Backend: `run.py` + `main.py`

### File: `backend/run.py`

| Line | Read aloud |
|------|------------|
| 1 | I import Uvicorn, which is the ASGI server that actually runs our FastAPI app. |
| 3 | This block only runs when I execute the file directly with `python run.py`. |
| 4 | I tell Uvicorn to load the app from `app.main:app` — that is the `app` object inside `main.py`. |
| 4 | Host `0.0.0.0` means it listens on all network interfaces; port `8000` is where the API lives. |
| 4 | `reload=True` auto-restarts the server when I change code during development. |

### File: `backend/app/main.py`

| Line | Read aloud |
|------|------------|
| 1–13 | The docstring lists every API group: auth, documents, Q&A, summaries, risks, evaluation, and features. |
| 16–17 | I import logging and `asynccontextmanager` for startup/shutdown hooks. |
| 19–20 | FastAPI builds the HTTP API; CORS middleware lets the React app on port 5173 call the backend. |
| 22 | I import all router modules — each file owns one feature area. |
| 23–25 | Settings, database init, and cleanup for expired documents. |
| 27–30 | I configure logging format so I can trace requests in the terminal. |
| 31 | I create a logger named `kanooni_sahayak`. |
| 34–35 | `lifespan` is an async context manager FastAPI runs once at startup and once at shutdown. |
| 36 | On startup I call `init_db()` — this creates SQLite tables if they do not exist. |
| 37–38 | I delete expired uploads and FAISS files — this is our data-minimization / privacy story. |
| 39 | I log that the API started. |
| 40 | `yield` means the app runs normally until shutdown. |
| 41 | On shutdown I log goodbye. |
| 44–45 | `create_app` builds and returns the FastAPI instance. |
| 46–50 | I set the API title, description, version, and attach the lifespan handler. |
| 52–57 | CORS: I read allowed origins from config — default is the Vite frontend URL. |
| 59–65 | I mount each router under `/api/...` with a URL prefix and Swagger tag. |
| 67–69 | A simple `/health` endpoint returns OK — useful for deployment checks. |
| 74 | `app = create_app()` — Uvicorn imports this object. |

---

## 2. `config.py` + `database.py`

### File: `backend/app/config.py`

| Line | Read aloud |
|------|------------|
| 1 | `lru_cache` means settings are loaded once and reused — efficient singleton. |
| 2 | `BaseSettings` reads values from environment variables and from the `.env` file. |
| 5–7 | `Settings` class: `model_config` points to `.env` and ignores unknown keys. |
| 8 | `google_api_key` — required for Gemini and Google embeddings. |
| 9 | `secret_key` — used to sign JWT tokens; must change in production. |
| 10 | `algorithm` — HS256 is the JWT signing algorithm. |
| 11–12 | Access tokens expire in 30 minutes; refresh tokens in 7 days. |
| 13 | `database_url` — SQLite with async driver `aiosqlite` for now. |
| 14 | `faiss_index_dir` — folder where per-document vector indexes are stored on disk. |
| 15 | `upload_dir` — where uploaded PDF/DOCX files are saved temporarily. |
| 16 | `document_ttl_hours` — after 24 hours we can delete files (privacy). |
| 17 | `classification_confidence_gap` — if top two class scores differ by less than 0.05, we ask the user to pick the type manually. |
| 18 | `gemini_model` — which Gemini model generates answers and summaries. |
| 19 | `embedding_model` — Google embedding model for vectors. |
| 20 | `reranker_model` — BGE cross-encoder for reranking retrieved chunks. |
| 21–23 | We retrieve 5 small chunks and 5 large chunks, then rerank down to top 8. |
| 24 | `cors_origins` — comma-separated list of allowed frontend URLs. |
| 26–28 | `cors_origin_list` splits that string into a Python list. |
| 31–33 | `get_settings()` returns a cached `Settings` instance. |

### File: `backend/app/database.py`

| Line | Read aloud |
|------|------------|
| 1–2 | SQLAlchemy async engine and session factory. |
| 4 | Import settings to get the database URL. |
| 6–7 | Create the async engine — `echo=False` means do not print every SQL statement. |
| 8 | `AsyncSessionLocal` is the factory for database sessions per request. |
| 11–12 | `Base` is the parent class for all ORM models (User, Document, Chunk, etc.). |
| 15–22 | `get_db` is a FastAPI dependency: open session, yield it to the route, commit on success, rollback on error. |
| 25–29 | `init_db` imports all models so SQLAlchemy knows the tables, then runs `create_all`. |

---

## 3. `api/documents.py` (upload endpoint)

### File: `backend/app/api/documents.py` — `upload_document` only

| Line | Read aloud |
|------|------------|
| 1–4 | Docstring: this file handles document CRUD, upload, and pipeline metadata for demos. |
| 23–30 | The upload endpoint accepts a file, optional language, optional forced document type, and requires a logged-in user. |
| 31–33 | I check the file extension — only `.pdf` and `.docx` are allowed. |
| 35–36 | I ensure the upload directory exists. |
| 37–38 | I build a unique filename with user id and UUID so files never collide. |
| 40–41 | I read the file bytes from the HTTP request and write them to disk. |
| 43–44 | I create `DocumentIngestionService` and call `ingest` — this runs the full AI pipeline. |
| 46–54 | I return JSON with document id, title, type, whether manual selection is needed, classification scores, and page count. |

**Say to interviewer:** “Upload does not just save the file — it immediately parses, classifies, chunks, embeds, and indexes in FAISS.”

---

## 4. `document_ingestion.py`

### File: `backend/app/services/document_ingestion.py`

| Line | Read aloud |
|------|------------|
| 1–6 | Module docstring: this orchestrates Phase 1 — everything that happens once per upload. |
| 8–12 | Standard imports: logging, datetime, pathlib, SQLAlchemy session. |
| 14–22 | I import config, database models, DocumentObject, chunker, classifier, embeddings, parser, FAISS store, PII redaction. |
| 24–25 | Logger and settings singleton. |
| 28 | `DocumentIngestionService` class — one place that wires all ingest steps. |
| 29–34 | In `__init__` I create the classifier, embedding service, FAISS store, and hybrid chunker. The chunker gets `embed_fn` so semantic fallback can embed sentences. |
| 36–44 | `ingest` method signature: database session, owner user id, file path, filename, language, and optional forced document type from the UI. |
| 45–46 | **Step 1:** `parse_document` reads PDF or DOCX into a `DocumentObject` with text, pages, metadata. |
| 47–48 | I run `redact_pii` on the full text — emails, phones, PAN, Aadhaar become placeholders. |
| 50–55 | **Step 2:** I classify the document. I pass `embed_fn` so we can optionally average segment embeddings. Gemini returns scores for each contract type. |
| 56 | `document_type` is either what the user forced on upload or what classification picked automatically. |
| 57–58 | `manual` is true when confidence gap was too small and the user did not force a type — UI will ask them to choose. |
| 60–65 | **Step 3:** Hybrid chunking. I pass `document_type` so structure chunker looks for type-specific headings like “Rent” for leases. |
| 62 | `chunk_document` returns a list of chunks plus metadata dict for the pipeline API. |
| 67–83 | I build `pipeline_meta` — a JSON blob stored on the document so `/pipeline` can prove what ran (classification, chunk counts, FAISS, reranker model name). |
| 85–86 | I set `expires_at` to now plus TTL hours — auto-delete for privacy. |
| 87–99 | I create the `Document` row in SQLite: title, path, type, scores, language, metadata, page count, expiry. |
| 100–101 | I add to DB and flush so we get `record.id` before indexing. |
| 103–105 | **Step 4:** I collect all chunk texts and embed them in batch asynchronously. |
| 107–108 | I prepare two lists: `stored` for FAISS metadata, `db_chunks` for SQLite. |
| 109 | I loop over each chunk and its embedding vector with index `i`. |
| 111–122 | For FAISS I build a `StoredChunk` with text, page, section, chunk size small/large, and character offsets for citations. |
| 125–137 | For SQLite I save a truncated chunk text (max 5000 chars) for debugging — search uses FAISS, not SQL. |
| 139 | `add_document` writes the FAISS index file and pickle metadata for this document id. |
| 140–141 | I save all chunk rows and flush again. |
| 142–143 | I log how many chunks were indexed and return the `Document` record. |

---

## 5. `hybrid_chunker.py`

### File: `backend/app/services/chunking/hybrid_chunker.py`

| Line | Read aloud |
|------|------------|
| 1–11 | Docstring explains we rejected fixed-only chunking; we use structure first, semantic fallback, and always both small and large sizes. |
| 22–23 | `SMALL_RANGE` is 200–400 tokens; `LARGE_RANGE` is 1000–1500 tokens. |
| 27–29 | Constructor stores optional `embed_fn` for semantic path. |
| 31–35 | `chunk_document` takes the parsed document and optional document type; returns chunks plus meta dict. |
| 36 | Empty list to collect all chunks. |
| 39–41 | I try structure chunking at legal headings for both small and large token windows. |
| 41 | `used_structure` is true if we found enough headings to split that way. |
| 44–47 | If structure worked for small chunks, I tag each chunk as `"small"` and add to the list. |
| 48–53 | Else I call `semantic_chunk` — splits when meaning shifts between sentences — still tagged small. |
| 56–59 | Same logic for large chunks with `chunk_size = "large"`. |
| 60–64 | Else semantic chunking for large windows. |
| 67–76 | I build `meta`: strategy name, whether structure was used, document type, token ranges, counts of small vs large chunks, total count. |
| 77–78 | Log and return chunks plus meta. |

---

## 6. `retriever.py`

### File: `backend/app/services/retriever.py`

| Line | Read aloud |
|------|------------|
| 1–9 | Docstring: hybrid retrieval — small chunks for precision, large for context; FAISS is fast but noisy so we rerank. |
| 22–26 | `HybridRetriever` holds FAISS store, embedding service, and reranker instances. |
| 28–35 | `retrieve` takes document id, question, and optional k values; returns ranked list of chunks with scores. |
| 36–38 | Default k values come from settings: 5 small, 5 large, rerank top 8. |
| 40–41 | I embed the user question using the `retrieval_query` task type — slightly different from document embeddings. |
| 44–45 | I search FAISS twice: once filtering `chunk_size="small"`, once `"large"`. |
| 47–51 | I merge results in a dictionary keyed by `chunk_id` — keep the higher score if the same chunk appeared twice. |
| 53–54 | I sort candidates by FAISS score descending. |
| 56–57 | I pass candidates to the BGE reranker, which reorders by true relevance of question plus chunk together. |

---

## 7. `rag_pipeline.py`

### File: `backend/app/services/rag_pipeline.py`

| Line | Read aloud |
|------|------------|
| 1–6 | Docstring: Phase 2 — user asks a question; we never send the full PDF to Gemini. |
| 22–26 | I configure Gemini API key and create the generative model and hybrid retriever. |
| 28–32 | `_build_context` turns ranked chunks into prompt text and citation objects. |
| 35 | I enumerate chunks starting at Source 1. |
| 36 | Each block gets a header `[Source N]`. |
| 38–41 | I append page number and section name when we have them from the parser. |
| 43 | The block contains the actual clause text. |
| 44–51 | I build a `CitationOut` with page, section, optional clause number, char offset, and a short excerpt. |
| 53 | I join all blocks with `---` separators for the prompt. |
| 55–61 | `ask` method: document id, question, user language, document language. |
| 62–63 | First I retrieve and rerank — see retriever.py. |
| 64–71 | If nothing was retrieved, I return the exact message “Answer not found in the document” with empty citations. |
| 73 | I build context string and citation list. |
| 75–78 | I normalize language codes and add cross-lingual instructions if question and document languages differ. |
| 79–83 | I format the system prompt and user prompt with context and question. |
| 85–87 | Gemini generates the answer. |
| 89–91 | If the answer is empty or looks like “not found”, I normalize to our standard phrase. |
| 93–95 | Confidence is derived from the top reranker score — shown in the UI. |
| 97–102 | I return answer, citations as dicts, confidence, and source labels. |

---

## 8. `api/qa.py`

### File: `backend/app/api/qa.py`

| Line | Read aloud |
|------|------------|
| 1–6 | Docstring: thin API over RAGPipeline; ChatPage calls this; we save conversation history. |
| 18 | FastAPI router for Q&A routes. |
| 21–26 | POST `/ask` expects JSON body, logged-in user, and database session. |
| 27–28 | I verify the user owns this document — authorization. |
| 30 | I instantiate `RAGPipeline`. |
| 31–36 | I call `pipeline.ask` with document id, question, user language, and document’s detected language. |
| 38–48 | I save a `Conversation` row with question, answer, language, confidence. |
| 50–60 | For each citation I save page, section, clause number, offset, excerpt linked to that conversation. |
| 62–67 | I return the response DTO the frontend expects: answer, citations, confidence, sources. |

---

## 9. Frontend: `ChatPage.jsx`

### File: `frontend/src/pages/ChatPage.jsx`

| Line | Read aloud |
|------|------------|
| 1–6 | Comment: this is the user-facing RAG chat; each send hits the full backend pipeline. |
| 7–9 | React imports: state hook, URL params, and our API helper. |
| 11 | Default export — the Chat page component. |
| 12 | `useParams` reads `:id` from the URL — which document we are chatting about. |
| 13 | `messages` holds the chat history in React state. |
| 14 | `question` is the current text in the input box. |
| 15 | `language` defaults to English — can switch to Hindi or Hinglish. |
| 16 | `loading` disables the button while waiting for the API. |
| 18–20 | `ask` runs on form submit; if question is empty, return. |
| 21 | Set loading true. |
| 23–24 | Call `qaApi.ask` with document id, question, language — this is POST `/api/qa/ask`. |
| 25–29 | Append user message and assistant response (answer, citations, confidence) to messages. |
| 30 | Clear the input. |
| 31–33 | Always turn off loading in `finally`. |
| 37–38 | Page title “Legal Q&A”. |
| 41–48 | Dropdown to pick answer language. |
| 51–77 | Map over messages: user bubbles on the right styling, assistant on the left with answer, confidence percent, and citation list with page and section. |
| 80–90 | Form with text input and Send button; disabled while loading. |

---

## 10. Frontend: `api.js`

### File: `frontend/src/services/api.js`

| Line | Read aloud |
|------|------------|
| 1–7 | Comment: all HTTP calls; Vite proxies `/api` to port 8000; JWT attached automatically. |
| 8 | Import Axios. |
| 10 | Create client with base URL `/api`. |
| 12–16 | Request interceptor: read `access_token` from localStorage and set `Authorization: Bearer ...` header. |
| 18–22 | `authApi`: register, login, get current user. |
| 24–26 | `featuresApi`: list implemented features for demos. |
| 28–39 | `documentsApi`: list, pipeline metadata, upload with FormData, get, delete, patch document type. |
| 41–44 | `qaApi.ask` posts document id, question, language — connects to RAG backend. |
| 46–48 | `summariesApi.generate` for contract summary. |
| 50–53 | `risksApi.analyze` and `get` for risk analysis. |
| 55–58 | `evaluationApi` for RAGAS runs and history. |

---

## 11. `legalDocumentAssistant/app.py` (prototype only)

**Say first:** “This is my early Streamlit prototype — not what I demo for the full architecture.”

| Section | Read aloud |
|---------|------------|
| Lines 1–12 | Docstring says this is the old proof-of-concept; full system is `kanooni-sahayak`. |
| 14–22 | Load Gemini API key and Google credentials from Streamlit secrets. |
| 25–27 | `extract_text_from_docx` — plain text from Word, no page numbers. |
| 29–40 | `get_document_text` loops uploaded files and concatenates PDF or DOCX text. |
| 42–45 | `get_text_chunks` uses fixed 10,000 character chunks — what we replaced with hybrid chunking. |
| 47–50 | `get_vector_store` embeds with Google and saves one shared FAISS folder. |
| 51–64 | `get_conversational_chain` — LangChain prompt: answer only from context. |
| 65–71 | `user_input` — similarity search, no reranker, no citations. |
| 73–108 | `summarize_document_dual` — one Gemini call, first 10k characters only. |
| 110–135 | `extract_risky_clauses` — prompt-only risks, no JSON frameworks. |
| 138–144 | Warning banner telling user to use Kanooni Sahayak for interviews. |
| 153–164 | On submit: extract text, chunk, FAISS, summarize — all in session state. |
| 166–186 | Three tabs: Summary, Risky Clauses, Ask a Question — same ideas, simpler pipeline. |

---

## Full story in 90 seconds (memorize this)

> “When the user uploads a contract, `documents.py` saves the file and calls `document_ingestion.py`. We parse with PyMuPDF or python-docx, redact PII, classify the contract type with a confidence-gap check, hybrid-chunk into small and large pieces, embed with Google, and index in FAISS per document.
>
> When they ask a question, `ChatPage` calls `qa.py`, which runs `rag_pipeline.py`. That embeds the question, retrieves five small and five large chunks from FAISS, merges them, reranks with BGE, and sends the top chunks to Gemini with a strict prompt. Answers include page citations. If the clause is not in the document, we say so explicitly.
>
> Risks use JSON frameworks plus open-ended discovery. Evaluation uses RAGAS. The old Streamlit app was the prototype; Kanooni Sahayak is the production design.”

---

## Other files (shorter pointers)

If asked to open these, say:

| File | One sentence |
|------|----------------|
| `vector_store.py` | Normalizes vectors, FAISS inner product equals cosine; one index file per document. |
| `reranker.py` | Lazy-loads BGE; scores `[question, chunk]` pairs; falls back to FAISS order on error. |
| `embedding_service.py` | Caches embeddings; `retrieval_document` vs `retrieval_query` task types. |
| `document_classifier.py` | Representative text + confidence gap; optional mean embedding hint to Gemini. |
| `risk_analyzer.py` | Loads `employment.json` etc.; framework pass then open-ended pass; dedupe. |
| `rag_prompt.py` | System rules: only context, exact not-found phrase, cite `[Source N]`. |
| `api/deps.py` | `get_current_user` validates JWT Bearer token. |
| `Dashboard.jsx` | Manual type dropdown when `manual_selection_required` is true. |
| `DocumentViewer.jsx` | Calls `/pipeline` and shows checkmarks for interview demo. |

---

*End of read-aloud script. Open the source file beside this document while practicing.*
