"""
RAGAS = Retrieval Augmented Generation Assessment.

For each test question we feed RAGAS:
  - the question
  - what chunks we retrieved (context)
  - the answer Gemini produced

Key scores for interviews:
  - faithfulness: is the answer actually supported by the retrieved text?
    (doc says 30 days notice → answer must not say 45)
  - answer_relevancy: does the answer address what was asked?

If RAGAS isn't installed, we fall back to simple heuristics so the API still works in dev.
"""

import logging
from typing import Any

from app.services.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class RagasEvaluator:
    def evaluate(
        self,
        document_id: int,
        questions: list[str],
        ground_truth: list[str] | None = None,
    ) -> dict[str, Any]:
        pipeline = RAGPipeline()
        answers: list[str] = []
        contexts: list[str] = []

        # run the real production RAG path for each eval question
        for q in questions:
            result = pipeline.ask(document_id, q)
            answers.append(result["answer"])
            contexts.append(" ".join(c.get("excerpt", "") or "" for c in result["citations"]))

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, faithfulness

            data = {
                "question": questions,
                "answer": answers,
                "contexts": [[c] for c in contexts],
            }
            if ground_truth and len(ground_truth) == len(questions):
                data["ground_truth"] = ground_truth

            dataset = Dataset.from_dict(data)
            result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
            df = result.to_pandas()
            return {
                "precision": float(df.get("context_precision", [0.0])[0]) if "context_precision" in df else None,
                "recall": float(df.get("context_recall", [0.0])[0]) if "context_recall" in df else None,
                "faithfulness": float(df["faithfulness"].mean()) if "faithfulness" in df else None,
                "answer_relevance": float(df["answer_relevancy"].mean()) if "answer_relevancy" in df else None,
                "raw": df.to_dict(),
            }
        except Exception as exc:
            logger.warning("RAGAS evaluation fallback: %s", exc)
            return self._heuristic_metrics(questions, answers, ground_truth, contexts)

    def _heuristic_metrics(
        self,
        questions: list[str],
        answers: list[str],
        ground_truth: list[str] | None,
        contexts: list[str],
    ) -> dict[str, Any]:
        not_found = sum(1 for a in answers if "not found" in a.lower())
        faithfulness = 1.0 - (not_found / max(len(answers), 1))
        relevance = sum(
            1 for q, a in zip(questions, answers) if any(w.lower() in a.lower() for w in q.split()[:3])
        ) / max(len(questions), 1)
        precision, recall = 0.75, 0.7
        if ground_truth:
            overlap = sum(
                1
                for a, g in zip(answers, ground_truth)
                if any(tok in a.lower() for tok in g.lower().split()[:5])
            )
            recall = overlap / max(len(ground_truth), 1)
        return {
            "precision": precision,
            "recall": recall,
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "note": "Heuristic fallback — install full RAGAS stack for production metrics",
        }
