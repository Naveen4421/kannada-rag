"""Query processing: the one place the user's question is cleaned.

The original text is always preserved and is what the generator sees. Only
`normalized_query` is used for embedding/BM25. Normalisation is deliberately
conservative: Unicode NFC, invisible-character and whitespace/punctuation
cleanup. No stemming or transliteration -- Kannada is agglutinative and the
index holds surface forms, so destructive normalisation would hurt recall.
"""
import re
import unicodedata
from dataclasses import dataclass, field, asdict

# Zero-width space / word joiner / BOM / soft hyphen: pure noise from
# copy-paste. ZWJ (U+200D) and ZWNJ (U+200C) are NOT removed: they control
# conjunct formation in Kannada (e.g. after virama) and change the word.
_NOISE = dict.fromkeys(map(ord, "​⁠﻿­"), None)
_PUNCT_MAP = {ord("？"): "?", ord("！"): "!", ord("‘"): "'", ord("’"): "'",
              ord("“"): '"', ord("”"): '"'}
_WS = re.compile(r"\s+")
_REPEAT_PUNCT = re.compile(r"([?!.,;:])\1+")

# Same marker idea as the router, but used only to label intent for logs and
# evidence handling; the router still owns simple/complex routing.
_INTENT_MARKERS = [
    ("comparison", ["ಹೋಲಿಸಿ", "ವ್ಯತ್ಯಾಸ", "compare"]),
    ("explanation", ["ಹೇಗೆ", "ಏಕೆ", "ಯಾಕೆ", "ವಿವರಿಸಿ", "why", "how"]),
    ("factoid", ["ಯಾರು", "ಎಲ್ಲಿ", "ಯಾವಾಗ", "ಎಷ್ಟು", "ಯಾವ", "who", "where", "when"]),
]


@dataclass
class ProcessedQuery:
    original_query: str
    normalized_query: str
    language: str
    intent: str
    book_filter: object = None
    metadata_filters: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def normalize_text(text):
    text = unicodedata.normalize("NFC", text or "")
    text = text.translate(_NOISE).translate(_PUNCT_MAP)
    text = _WS.sub(" ", text).strip()
    return _REPEAT_PUNCT.sub(r"\1", text)


def detect_language(text):
    kannada = sum(1 for ch in text if "ಀ" <= ch <= "೿")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    letters = kannada + latin

    if letters == 0:
        return "unknown"
    if kannada / letters >= 0.8:
        return "kn"
    if latin / letters >= 0.8:
        return "en"
    return "mixed"


def detect_intent(text):
    lowered = text.lower()
    for intent, markers in _INTENT_MARKERS:
        if any(marker in lowered for marker in markers):
            return intent
    return "general"


def process_query(question, where=None):
    where = dict(where) if where else {}
    normalized = normalize_text(question)

    return ProcessedQuery(
        original_query=question,
        normalized_query=normalized,
        language=detect_language(normalized),
        intent=detect_intent(normalized),
        book_filter=where.get("book_id"),
        metadata_filters=where,
    )
