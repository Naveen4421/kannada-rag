import numpy as np

from retrieval.routing import router


def test_router_diagnostics(monkeypatch):
    refs = {"simple": np.array([[1.0, 0.0]]), "complex": np.array([[0.0, 1.0]])}
    monkeypatch.setattr(router, "_reference_embeddings", lambda: refs)

    d = router.classify("ಇದು ಏಕೆ ಮುಖ್ಯ? ವಿವರಿಸಿ", np.array([0.0, 1.0]))
    assert d.route == "complex" and d.use_self_critique and d.top_k == 15
    assert d.heuristic_score == 1.0 and d.semantic_score == 1.0
    assert d.combined_score == d.score == 1.0 and d.confidence == 1.0
    assert "complex=['ಏಕೆ', 'ವಿವರಿಸಿ']" in d.reason

    d = router.classify("ಲೇಖಕರು ಯಾರು?", np.array([1.0, 0.0]))
    assert d.route == "simple" and not d.use_self_critique and d.top_n == 3
    assert d.combined_score < 0 and set(d.diagnostics()) >= {"heuristic_score", "semantic_score", "reason"}


def test_no_marker_defers_to_embedding(monkeypatch):
    monkeypatch.setattr(router, "_reference_embeddings",
                        lambda: {"simple": np.array([[1.0, 0.0]]), "complex": np.array([[0.0, 1.0]])})
    d = router.classify("xyz", np.array([0.05, 0.0]))
    assert d.heuristic_score == 0.0 and d.combined_score == d.semantic_score and "no question-word marker" in d.reason
