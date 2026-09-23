from dataclasses import dataclass
from functools import lru_cache

import numpy as np


# Kannada is agglutinative — question words take suffixes/sandhi
# (ಎಲ್ಲಿ -> ಎಲ್ಲಿಂದ, ಯಾವಾಗ -> ಯಾವಾಗಲೂ) so markers are matched as
# substrings, not exact tokens.
SIMPLE_MARKERS = ["ಎಲ್ಲಿ", "ಯಾವಾಗ", "ಯಾರು", "ಎಷ್ಟು", "ಯಾವ"]
COMPLEX_MARKERS = ["ಹೇಗೆ", "ಏಕೆ", "ಯಾಕೆ", "ವಿವರಿಸಿ", "ಹೋಲಿಸಿ", "ವಿಶೇಷತೆ"]

REFERENCE_QUESTIONS = {
    "simple": [
        "ಅವರು ಎಲ್ಲಿ ಜನಿಸಿದರು?",
        "ಈ ಪುಸ್ತಕ ಯಾವಾಗ ಪ್ರಕಟವಾಯಿತು?",
        "ಲೇಖಕರು ಯಾರು?",
        "ಒಟ್ಟು ಎಷ್ಟು ಪುಟಗಳಿವೆ?",
        "ಅವರ ತಂದೆಯ ಹೆಸರೇನು?",
    ],
    "complex": [
        "ಅವರ ಸಂಶೋಧನಾ ಕಾರ್ಯದ ವಿಶೇಷತೆ ಏನು?",
        "ಅವರು ಈ ಕೆಲಸವನ್ನು ಹೇಗೆ ಸಾಧಿಸಿದರು ಎಂದು ವಿವರಿಸಿ.",
        "ಎರಡು ಪುಸ್ತಕಗಳ ದೃಷ್ಟಿಕೋನಗಳನ್ನು ಹೋಲಿಸಿ.",
        "ಅವರ ಸ್ವಭಾವ ಮತ್ತು ಕಾರ್ಯವೈಖರಿ ಹೇಗಿತ್ತು?",
        "ಇದು ಏಕೆ ಮುಖ್ಯವಾಗಿತ್ತು ಎಂದು ವಿವರಿಸಿ.",
    ],
}


@dataclass
class RouteDecision:
    route: str
    score: float
    top_k: int
    top_n: int
    use_query_expansion: bool
    use_self_critique: bool


@lru_cache(maxsize=1)
def _reference_embeddings():
    # Imported lazily so this module has no hard import-time dependency on
    # the embedding stack, and to avoid a circular import with
    # src.embeddings.model at module load time.
    from src.embeddings.model import embed_texts

    return {
        label: embed_texts(questions)
        for label, questions in REFERENCE_QUESTIONS.items()
    }


def _heuristic_score(question):
    simple_hits = sum(1 for marker in SIMPLE_MARKERS if marker in question)
    complex_hits = sum(1 for marker in COMPLEX_MARKERS if marker in question)

    if simple_hits == 0 and complex_hits == 0:
        return 0.0

    total = simple_hits + complex_hits
    return (complex_hits - simple_hits) / total


def _embedding_score(question_embedding):
    refs = _reference_embeddings()

    simple_sim = float(np.mean(np.dot(refs["simple"], question_embedding)))
    complex_sim = float(np.mean(np.dot(refs["complex"], question_embedding)))

    return complex_sim - simple_sim


def classify(question, question_embedding):
    heuristic = _heuristic_score(question)
    embedding = _embedding_score(question_embedding)

    # A Kannada question with no marker hit at all is a weak/noisy heuristic
    # signal (agglutination means absence of a listed marker proves little)
    # -- defer entirely to the embedding signal rather than letting a
    # heuristic score of 0 silently pull the combination toward "simple".
    if heuristic == 0.0:
        combined = embedding
    else:
        combined = 0.4 * heuristic + 0.6 * embedding

    route = "complex" if combined > 0 else "simple"

    if route == "complex":
        return RouteDecision(
            route=route,
            score=combined,
            top_k=15,
            top_n=6,
            use_query_expansion=True,
            use_self_critique=True,
        )

    return RouteDecision(
        route=route,
        score=combined,
        top_k=6,
        top_n=3,
        use_query_expansion=False,
        use_self_critique=False,
    )
