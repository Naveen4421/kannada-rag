from common.llm import generate_answer


EXTRACTION_PROMPT = (
    "ಈ ಕೆಳಗಿನದು ಒಂದು ಪುಸ್ತಕದ ಆರಂಭಿಕ ಪುಟಗಳ ಪಠ್ಯ. ಈ ಪುಸ್ತಕದ ಶೀರ್ಷಿಕೆ ಮತ್ತು "
    "ಲೇಖಕರ ಹೆಸರನ್ನು ಗುರುತಿಸಿ. ನಿಖರವಾಗಿ ಈ ಸ್ವರೂಪದಲ್ಲಿ ಮಾತ್ರ ಉತ್ತರಿಸಿ, ಬೇರೆ ಏನೂ "
    "ಬರೆಯಬೇಡಿ. ಪಠ್ಯದಲ್ಲಿ ಒಂದು ಅಂಶ ಕಂಡುಬರದಿದ್ದರೆ ಆ ಸಾಲಿನಲ್ಲಿ 'ಗೊತ್ತಿಲ್ಲ' ಎಂದು ಬರೆಯಿರಿ:\n"
    "ಶೀರ್ಷಿಕೆ: <ಶೀರ್ಷಿಕೆ ಅಥವಾ ಗೊತ್ತಿಲ್ಲ>\n"
    "ಲೇಖಕರು: <ಲೇಖಕರ ಹೆಸರು ಅಥವಾ ಗೊತ್ತಿಲ್ಲ>\n\n"
    "ಪಠ್ಯ:\n{text}"
)

MAX_SAMPLE_CHARS = 1500


def _sample_opening_text(doc, max_chars=MAX_SAMPLE_CHARS):
    pieces = []
    total = 0

    for page in doc["pages"]:
        for paragraph in page["paragraphs"]:
            pieces.append(paragraph["text"])
            total += len(paragraph["text"])

            if total >= max_chars:
                return "\n".join(pieces)

    return "\n".join(pieces)


def _parse_metadata(raw):
    title = None
    author = None

    for line in raw.strip().split("\n"):
        line = line.strip()

        if line.startswith("ಶೀರ್ಷಿಕೆ:"):
            value = line[len("ಶೀರ್ಷಿಕೆ:"):].strip()
            title = value if value and "ಗೊತ್ತಿಲ್ಲ" not in value else None
        elif line.startswith("ಲೇಖಕರು:"):
            value = line[len("ಲೇಖಕರು:"):].strip()
            author = value if value and "ಗೊತ್ತಿಲ್ಲ" not in value else None

    return title, author


def extract_metadata(doc):
    """
    Best-effort title/author extraction from the book's own opening pages.
    Returns (title, author), either of which may be None if not found or if
    the LLM call itself fails (e.g. no credits) -- callers should fall back
    to filename-derived defaults, never block ingestion on this.
    """
    sample = _sample_opening_text(doc)

    if not sample.strip():
        return None, None

    prompt = EXTRACTION_PROMPT.format(text=sample)

    try:
        raw = generate_answer(prompt)
    except Exception:
        return None, None

    return _parse_metadata(raw)
