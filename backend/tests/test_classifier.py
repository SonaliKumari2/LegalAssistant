from app.services.document_classifier import DocumentClassifier, _heuristic_scores


def test_heuristic_scores_sum_approx_one():
    scores = _heuristic_scores("This employment agreement between employee and employer includes salary and probation.")
    assert abs(sum(scores.values()) - 1.0) < 0.01
    assert scores["Employment Contract"] > scores["NDA"]


def test_confidence_gap_manual_selection():
    clf = DocumentClassifier(confidence_gap=0.5)
    result = clf.classify("generic legal text without strong signals " * 20)
    assert "manual_selection_required" in result
    assert "classification_scores" in result
