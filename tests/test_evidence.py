import pytest

from retrieval.config import EvidenceConfig
from retrieval.evidence import agreement, citations, scorer
from retrieval.evidence.engine import NO_EVIDENCE, WEAK_EVIDENCE, LOW_RERANK_CONFIDENCE, build_evidence, assess_sufficiency
from tests.conftest import make_chunk

BORN = "ರಾಜಪುರೋಹಿತರು 1880 ರಲ್ಲಿ ಧಾರವಾಡದಲ್ಲಿ ಜನಿಸಿದರು ಎಂದು ತಿಳಿದುಬರುತ್ತದೆ"
BORN_B = "ರಾಜಪುರೋಹಿತರು ಧಾರವಾಡದಲ್ಲಿ 1880 ರಲ್ಲಿ ಜನಿಸಿದರು ಎಂದು ಹೇಳಲಾಗಿದೆ"
BORN_WRONG = "ರಾಜಪುರೋಹಿತರು 1885 ರಲ್ಲಿ ಧಾರವಾಡದಲ್ಲಿ ಜನಿಸಿದರು ಎಂದು ತಿಳಿದುಬರುತ್ತದೆ"
UNRELATED = "ಕುವೆಂಪು ಕುಪ್ಪಳ್ಳಿಯಲ್ಲಿ ಹುಟ್ಟಿ ಕವಿತೆಗಳನ್ನು ಬರೆದರು ಮತ್ತು ನಾಟಕ ರಚಿಸಿದರು"


def chunk(cid, book, text, rerank=0.9, page=5, **kw):
    c = make_chunk(cid, book, text, page=page, title=f"Title {book}")
    c.update(rrf_score=0.03, dense_rank=1, sparse_rank=1, rerank_score=rerank, rerank_score_norm=rerank)
    c.update(kw)
    return c


def items_for(*chunks):
    return build_evidence(list(chunks)).items


def test_evidence_preserves_source_metadata_and_ids():
    bundle = build_evidence([chunk("A_1", "A", BORN, page=42), chunk("B_1", "B", UNRELATED, page=7)])
    first = bundle.items[0]
    assert (first["evidence_id"], first["book_id"], first["book_name"], first["page"], first["chunk_id"]) == \
        ("E1", "A", "Title A", "42", "A_1")
    assert bundle.items[1]["evidence_id"] == "E2"


def test_score_components_visible_and_weighted():
    item = items_for(chunk("A_1", "A", BORN))[0]
    comp = item["components"]
    assert set(comp) == {"retrieval", "reranker", "source_quality", "agreement"}
    assert 0.0 <= item["evidence_score"] <= 1.0
    assert sum(item["weights_used"].values()) == pytest.approx(1.0, abs=1e-3)


def test_missing_reranker_renormalises_instead_of_imputing():
    item = items_for(chunk("A_1", "A", BORN, rerank=None, rerank_score_norm=None))[0]
    assert item["components"]["reranker"] is None and "reranker" not in item["weights_used"]


def test_higher_reranker_gives_higher_score():
    hi, lo = items_for(chunk("A_1", "A", BORN, rerank=0.95), chunk("B_1", "B", UNRELATED, rerank=0.1))
    assert hi["evidence_score"] > lo["evidence_score"]


def test_independent_books_corroborate():
    items = items_for(chunk("A_1", "A", BORN), chunk("B_1", "B", BORN_B))
    rep = agreement.analyze(items)
    assert rep["status"] == "corroborated"
    assert rep["per_item"]["E1"]["independent_sources"] == ["B"]
    assert items[0]["components"]["agreement"] > 0.5


def test_same_book_passages_are_not_agreement():
    items = items_for(chunk("A_1", "A", BORN), chunk("A_2", "A", BORN_B))
    rep = agreement.analyze(items)
    assert rep["status"] == "single_source"
    assert rep["per_item"]["E1"]["same_book_passages"] == 1
    assert rep["per_item"]["E1"]["independent_sources"] == []


def test_same_book_duplicate_is_flagged_and_not_counted():
    items = items_for(chunk("A_1", "A", BORN), chunk("A_2", "A", BORN))
    info = agreement.analyze(items)["per_item"]["E1"]
    assert info["near_duplicate_of"] == ["E2"] and info["same_book_passages"] == 0


def test_conflicting_years_across_books():
    items = items_for(chunk("A_1", "A", BORN), chunk("B_1", "B", BORN_WRONG), chunk("C_1", "C", BORN_B))
    rep = agreement.analyze(items)
    assert rep["status"] == "conflicting"
    assert {tuple(c["evidence_ids"]) for c in rep["conflicts"]} == {("E1", "E2"), ("E2", "E3")}
    assert items[1]["components"]["agreement"] == 0.2


def test_unrelated_books_do_not_agree_or_conflict():
    assert agreement.analyze(items_for(chunk("A_1", "A", BORN), chunk("B_1", "B", UNRELATED)))["status"] == "single_source"


def test_sufficiency_codes():
    assert assess_sufficiency([])["reason"] == NO_EVIDENCE
    low = items_for(chunk("A_1", "A", BORN, rerank=0.01))
    assert assess_sufficiency(low)["reason"] == LOW_RERANK_CONFIDENCE
    weak = items_for(chunk("A_1", "A", BORN, rerank=0.2, rrf_score=0.001))
    assert assess_sufficiency(weak, EvidenceConfig(strong_score=0.9))["reason"] == WEAK_EVIDENCE
    assert assess_sufficiency(items_for(chunk("A_1", "A", BORN, rerank=0.95)))["sufficient"]


def test_source_quality_penalises_garbage_text_and_missing_metadata():
    good = chunk("A_1", "A", BORN, author="Real Author")
    junk = chunk("F_1", "F", "TM tax.com 24ColorCard Camer آرارارارارار 7 8 10", title="F", author="Unknown")
    assert scorer.source_quality_component(good) > scorer.source_quality_component(junk) + 0.4
    assert scorer.text_quality({"ocr_quality": 0.3, "text": BORN}) == 0.3  # explicit OCR quality wins


def test_citation_extraction_and_validation():
    items = items_for(chunk("A_1", "A", BORN, page=42), chunk("B_1", "B", UNRELATED, page=7))
    ok = citations.validate_citations("ಉತ್ತರ [E1, E2] ಪುಟ ೪೨ ರಲ್ಲಿ", items)
    assert ok["valid"] and ok["cited_ids"] == ["E1", "E2"] and ok["mentioned_pages"] == [42]
    bad = citations.validate_citations("ಉತ್ತರ [E9] page 99", items)
    assert bad["invalid_ids"] == ["E9"] and bad["invalid_pages"] == [99] and not bad["valid"]
    assert not citations.validate_citations("no citations here", items)["has_citations"]


def test_page_mention_forms():
    assert citations.extract_page_mentions("pp. 3-5 and 42ನೇ ಪುಟ") == [3, 4, 5, 42]
