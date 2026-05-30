"""Feature manifest for demos — maps interview talking points to implementation."""

from fastapi import APIRouter

router = APIRouter()

FEATURES = [
    {
        "id": 1,
        "name": "Document Ingestion",
        "description": "PDF (PyMuPDF) and DOCX (python-docx) with text, pages, metadata, headings",
        "implemented": True,
        "files": ["services/parsers/pdf_parser.py", "services/parsers/docx_parser.py"],
    },
    {
        "id": 2,
        "name": "Document Classification",
        "description": "7 types + confidence gap → manual selection if ambiguous",
        "implemented": True,
        "files": ["services/document_classifier.py", "utils/text_sampling.py"],
    },
    {
        "id": 3,
        "name": "Hybrid Chunking",
        "description": "Structure-first (type-specific headings), semantic fallback; small 200-400 & large 1000-1500 tokens",
        "implemented": True,
        "files": ["services/chunking/hybrid_chunker.py", "services/chunking/structure_chunker.py", "services/chunking/semantic_chunker.py"],
    },
    {
        "id": 4,
        "name": "Google Embeddings",
        "description": "Batch, cache, retry, async; multilingual tradeoff vs Legal-BERT",
        "implemented": True,
        "files": ["services/embedding_service.py"],
    },
    {
        "id": 5,
        "name": "FAISS Vector Index",
        "description": "Per-document index, cosine via L2-normalized inner product, local/privacy",
        "implemented": True,
        "files": ["services/vector_store.py"],
    },
    {
        "id": 6,
        "name": "Hybrid Retrieval + BGE Reranker",
        "description": "Top-k small + top-k large → merge → cross-encoder rerank (question+chunk)",
        "implemented": True,
        "files": ["services/retriever.py", "services/reranker.py"],
    },
    {
        "id": 7,
        "name": "RAG Q&A with Citations",
        "description": "Gemini grounded on context; 'Answer not found in the document.'",
        "implemented": True,
        "files": ["services/rag_pipeline.py", "prompts/rag_prompt.py"],
    },
    {
        "id": 8,
        "name": "Contract Summarization",
        "description": "Executive summary, obligations, dates, payment, termination, risks, concerns (EN/HI)",
        "implemented": True,
        "files": ["services/summarizer.py"],
    },
    {
        "id": 9,
        "name": "Risky Clause Extraction",
        "description": "JSON frameworks + open-ended discovery; severity Low/Medium/High",
        "implemented": True,
        "files": ["services/risk_analyzer.py", "risk_frameworks/*.json"],
    },
    {
        "id": 10,
        "name": "Multilingual",
        "description": "English, Hindi, Hinglish cross-lingual Q&A",
        "implemented": True,
        "files": ["utils/language.py"],
    },
    {
        "id": 11,
        "name": "Evaluation (RAGAS + Risk P/R)",
        "description": "Faithfulness, answer relevance, precision/recall on validation set",
        "implemented": True,
        "files": ["evaluation/ragas_evaluator.py", "evaluation/risk_metrics.py"],
    },
    {
        "id": 12,
        "name": "Privacy & Security",
        "description": "JWT, per-user docs, PII redaction, TTL cleanup",
        "implemented": True,
        "files": ["utils/security.py", "utils/pii_redaction.py", "utils/cleanup_expired.py"],
    },
]


@router.get("/")
def list_features():
    return {
        "product": "Kanooni Sahayak",
        "tagline": "AI-Powered Legal Assistant — RAG over contracts",
        "rag_definition": (
            "Instead of answering only from LLM training data, retrieve relevant passages "
            "from the uploaded contract, then generate an answer grounded in that context."
        ),
        "features": FEATURES,
        "all_implemented": all(f["implemented"] for f in FEATURES),
    }
