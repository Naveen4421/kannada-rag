"""Evidence IDs and citation handling.

Source metadata (book, page, chunk) is attached by the pipeline from the
retrieved payload and never taken from the model. The model may only refer to
evidence by its ID (`[E1]`); the validator checks that every ID and every
page number the answer mentions actually exists in the supplied evidence.
"""
import re

_KN_DIGITS = str.maketrans("೦೧೨೩೪೫೬೭೮೯", "0123456789")
_BRACKET = re.compile(r"\[([^\[\]]*)\]")
_EID = re.compile(r"\bE(\d+)\b")
_PAGE_AFTER = re.compile(r"(?:pages?|pp?\.?|ಪುಟಗಳು|ಪುಟ)\s*(\d+)(?:\s*[-–]\s*(\d+))?", re.IGNORECASE)
_PAGE_BEFORE = re.compile(r"(\d+)\s*(?:ನೇ|ನೆಯ|ರ)?\s*ಪುಟ")


def to_ascii_digits(text):
    return (text or "").translate(_KN_DIGITS)


def make_evidence_id(index):
    """1-based position -> 'E1'."""
    return f"E{index}"


def page_label(item):
    start, end = item.get("page_start"), item.get("page_end")
    if start is None:
        return "unknown"
    return str(start) if start == end or end is None else f"{start}-{end}"


def format_citation(item):
    return f"[{item['evidence_id']}] {item['book_name']}, p. {page_label(item)}"


def extract_cited_ids(answer):
    """Evidence IDs the answer cites, in order of first appearance. Only
    bracketed citations count ('[E1]', '[E1, E3]')."""
    seen = []
    for group in _BRACKET.findall(answer or ""):
        for match in _EID.finditer(group):
            eid = f"E{match.group(1)}"
            if eid not in seen:
                seen.append(eid)
    return seen


def extract_page_mentions(answer):
    """Page numbers the answer states (Kannada or ASCII digits, ranges expanded)."""
    text = to_ascii_digits(answer)
    pages = set()

    for match in _PAGE_AFTER.finditer(text):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if end >= start and end - start <= 50:
            pages.update(range(start, end + 1))
        else:
            pages.add(start)

    for match in _PAGE_BEFORE.finditer(text):
        pages.add(int(match.group(1)))

    return sorted(pages)


def evidence_pages(items):
    pages = set()
    for item in items:
        start, end = item.get("page_start"), item.get("page_end")
        if start is None:
            continue
        pages.update(range(int(start), int(end if end is not None else start) + 1))
    return pages


def validate_citations(answer, items):
    known = {item["evidence_id"] for item in items}
    cited = extract_cited_ids(answer)
    invalid_ids = [eid for eid in cited if eid not in known]

    mentioned = extract_page_mentions(answer)
    allowed = evidence_pages(items)
    invalid_pages = [p for p in mentioned if p not in allowed]

    return {
        "cited_ids": cited,
        "invalid_ids": invalid_ids,
        "mentioned_pages": mentioned,
        "invalid_pages": invalid_pages,
        "has_citations": bool(cited),
        "valid": not invalid_ids and not invalid_pages,
    }
