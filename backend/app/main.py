"""
Kanooni Sahayak API entry point.

Wires together:
  /api/auth      — register, login, JWT
  /api/documents — upload, list, pipeline metadata
  /api/qa        — RAG question answering
  /api/summaries — contract summaries
  /api/risks     — risky clause extraction
  /api/evaluation— RAGAS metrics
  /api/features  — list what's implemented (good for interview walkthrough)

Run: python run.py  →  http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, documents, evaluation, features, qa, risks, summaries
from app.config import get_settings
from app.database import init_db
from app.utils.cleanup_expired import cleanup_expired_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("kanooni_sahayak")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    # privacy: drop expired uploads + FAISS files on startup
    await cleanup_expired_documents()
    logger.info("Kanooni Sahayak API started")
    yield
    logger.info("Kanooni Sahayak API shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Kanooni Sahayak",
        description="AI Legal Assistant — RAG over contracts with hybrid retrieval + reranking",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
    app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
    app.include_router(summaries.router, prefix="/api/summaries", tags=["Summaries"])
    app.include_router(qa.router, prefix="/api/qa", tags=["Question Answering"])
    app.include_router(risks.router, prefix="/api/risks", tags=["Risk Analysis"])
    app.include_router(evaluation.router, prefix="/api/evaluation", tags=["Evaluation"])
    app.include_router(features.router, prefix="/api/features", tags=["Features"])

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "kanooni-sahayak"}

    return app


app = create_app()
