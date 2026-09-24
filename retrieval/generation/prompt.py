SYSTEM_INSTRUCTIONS = (
    "ನೀವು ಕನ್ನಡ ಪುಸ್ತಕಗಳ ಆಧಾರದ ಮೇಲೆ ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರಿಸುವ ಸಹಾಯಕರು.\n"
    "ಕೆಳಗೆ ನೀಡಿರುವ ಸಂದರ್ಭದ (context) ಮಾಹಿತಿಯನ್ನು ಮಾತ್ರ ಬಳಸಿ ಉತ್ತರಿಸಿ.\n"
    "ಪ್ರಶ್ನೆಯು ನಿರ್ದಿಷ್ಟವಾಗಿ ಕೇಳುತ್ತಿರುವುದನ್ನೇ (ಯಾರು/ಏನು/ಎಲ್ಲಿ/ಯಾವಾಗ/ಏಕೆ) "
    "ಗಮನಿಸಿ ಉತ್ತರಿಸಿ. ಸಂದರ್ಭದಲ್ಲಿ ಸಂಬಂಧಿತ ಮಾಹಿತಿ ಇದ್ದರೂ, ಪ್ರಶ್ನೆ ಕೇಳಿದ "
    "ನಿರ್ದಿಷ್ಟ ಅಂಶವೇ (ಉದಾ: ಸ್ಥಳ) ಸಂದರ್ಭದಲ್ಲಿ ಇಲ್ಲದಿದ್ದರೆ, ಬೇರೆ ಸಂಬಂಧಿತ "
    "ಸಂಗತಿಯನ್ನು (ಉದಾ: ದಿನಾಂಕ) ಬದಲಿ ಉತ್ತರವಾಗಿ ನೀಡಬೇಡಿ — ಬದಲಿಗೆ ಆ ನಿರ್ದಿಷ್ಟ "
    "ಅಂಶ ಸಂದರ್ಭದಲ್ಲಿ ಸ್ಪಷ್ಟವಾಗಿ ಇಲ್ಲ ಎಂದು ಹೇಳಿ, ನಂತರ ಸಂದರ್ಭದಲ್ಲಿ ಸಿಗುವ "
    "ಪೂರಕ ಮಾಹಿತಿಯನ್ನು (ಇದ್ದರೆ) ಪ್ರತ್ಯೇಕವಾಗಿ ಸೂಚಿಸಬಹುದು.\n"
    "ಉತ್ತರದ ಜೊತೆಗೆ ಆಧಾರವಾದ ಮೂಲ (ಪುಸ್ತಕ, ಲೇಖಕ, ಪುಟ, ಚಂಕ್) ಅನ್ನು ಉಲ್ಲೇಖಿಸಿ."
)


def format_context(chunks):
    blocks = []

    for number, chunk in enumerate(chunks, start=1):
        header = (
            f"[Source {number}: {chunk['title']} by {chunk['author']}, "
            f"page {chunk['page_start']}-{chunk['page_end']}, "
            f"chunk {chunk['chunk_id']}]"
        )

        blocks.append(f"{header}\n{chunk['text']}")

    return "\n\n".join(blocks)


def build_prompt(question, chunks):
    context = format_context(chunks)

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"--- Context ---\n{context}\n\n"
        f"--- Question ---\n{question}\n\n"
        f"--- Answer (in Kannada) ---"
    )
