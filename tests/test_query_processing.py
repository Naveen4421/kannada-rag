import unicodedata

from src.query_processing.processor import detect_intent, detect_language, normalize_text, process_query


def test_original_is_preserved_and_normalized_is_separate():
    raw = "  ರಾಜಪುರೋಹಿತರು​   ಎಲ್ಲಿ ಜನಿಸಿದರು？？  "
    pq = process_query(raw, where={"book_id": "TK003"})
    assert pq.original_query == raw
    assert pq.normalized_query == "ರಾಜಪುರೋಹಿತರು ಎಲ್ಲಿ ಜನಿಸಿದರು?"
    assert (pq.language, pq.intent, pq.book_filter, pq.metadata_filters) == ("kn", "factoid", "TK003", {"book_id": "TK003"})
    assert set(pq.to_dict()) == {"original_query", "normalized_query", "language", "intent", "book_filter", "metadata_filters"}


def test_unicode_nfc_normalization():
    decomposed = unicodedata.normalize("NFD", "ಕ್ಷೇತ್ರ")
    assert normalize_text(decomposed) == unicodedata.normalize("NFC", "ಕ್ಷೇತ್ರ")


def test_zwj_zwnj_are_kept_they_change_kannada_words():
    word = "ಕ್‍ಕ"  # virama + ZWJ
    assert normalize_text(word) == word
    assert "​" not in normalize_text("ಅ​ಆ") and normalize_text("ಅ​ಆ") == "ಅಆ"


def test_no_destructive_changes_to_kannada_morphology_digits_or_case():
    text = "ಸ್ಥಳಗಳಿಂದ ೧೯೦೪ ರಲ್ಲಿ Kuvempu"
    assert normalize_text(text) == text  # no stemming, no digit conversion, no lowercasing


def test_whitespace_and_punctuation_cleanup():
    assert normalize_text("ಯಾರು?!\n\t ಅವರು") == "ಯಾರು?! ಅವರು"
    assert normalize_text("ಯಾರು???") == "ಯಾರು?"
    assert normalize_text("a “b”") == 'a "b"'
    assert normalize_text(None) == ""


def test_language_and_intent():
    assert detect_language("ಕನ್ನಡ ಪ್ರಶ್ನೆ") == "kn" and detect_language("what is this") == "en"
    assert detect_language("ಕನ್ನಡ mixed words here also") == "mixed" and detect_language("123 ?") == "unknown"
    assert detect_intent("ಎರಡು ಪುಸ್ತಕಗಳನ್ನು ಹೋಲಿಸಿ") == "comparison"
    assert detect_intent("ಇದು ಏಕೆ ಮುಖ್ಯ") == "explanation" and detect_intent("ಲೇಖಕರ ಹೆಸರು") == "general"


def test_no_filter_gives_empty_metadata():
    pq = process_query("ಪ್ರಶ್ನೆ")
    assert pq.book_filter is None and pq.metadata_filters == {}
