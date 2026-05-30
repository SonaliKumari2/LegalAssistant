"""
Measure risky-clause extraction quality — precision & recall vs manual labels.

Precision = of what we flagged, how many were actually risky?
Recall    = of all risky clauses in the doc, how many did we catch?

Interview line: recall is more important — missing a bad clause is worse than over-flagging.

Example gold labels JSON:
[
  {"document_id": 1, "clause_substrings": ["terminate without cause", "unlimited liability"]},
]
"""

from typing import Any


def clause_match(predicted: str, gold_substrings: list[str]) -> bool:
    """Fuzzy match — clause text doesn't have to be character-perfect."""
    p = predicted.lower()
    return any(g.lower() in p or p in g.lower() for g in gold_substrings)


def risk_precision_recall(
    predicted_clauses: list[str],
    gold_substrings: list[str],
) -> dict[str, float]:
    if not predicted_clauses and not gold_substrings:
        return {"precision": 1.0, "recall": 1.0}
    if not predicted_clauses:
        return {"precision": 0.0, "recall": 0.0}
    if not gold_substrings:
        return {"precision": 0.0, "recall": 1.0}

    tp = sum(1 for p in predicted_clauses if clause_match(p, gold_substrings))
    precision = tp / len(predicted_clauses)
    recall = tp / len(gold_substrings)
    return {"precision": precision, "recall": recall}


def aggregate_metrics(items: list[dict[str, Any]]) -> dict[str, float]:
    if not items:
        return {"precision": 0.0, "recall": 0.0, "count": 0}
    p = sum(i["precision"] for i in items) / len(items)
    r = sum(i["recall"] for i in items) / len(items)
    return {"precision": p, "recall": r, "count": len(items)}
