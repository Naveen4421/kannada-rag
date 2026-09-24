from pathlib import Path

import streamlit as st

from ingestion.ingest import ingest_file
from common.catalog import list_books


st.set_page_config(page_title="Ingest — Kannada RAG", page_icon="📥")

st.title("📥 Ingest a book")
st.caption(
    "Upload a .docx or .txt file. Title and author are read automatically "
    "from the book's own opening pages — no need to type them in."
)

INPUT_DIR = Path("data/input")

uploaded = st.file_uploader("Book file", type=["docx", "txt"])
ingest_clicked = st.button("Ingest", type="primary")

if ingest_clicked:
    if uploaded is None:
        st.warning("Please choose a file first.")
    else:
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        dest = INPUT_DIR / uploaded.name

        with dest.open("wb") as f:
            f.write(uploaded.getbuffer())

        try:
            with st.spinner("Loading -> reading title/author -> chunking -> embedding -> indexing..."):
                summary = ingest_file(dest)
        except NotImplementedError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")
        else:
            st.success(
                f"Indexed **{summary['title']}** by {summary['author']} "
                f"({summary['book_id']}) — {summary['pages']} pages, "
                f"{summary['paragraphs']} paragraphs, {summary['chunks']} chunks."
            )

st.divider()
st.subheader("Currently indexed books")
for b in list_books():
    st.caption(f"- {b['title']} ({b['book_id']}) — {b['author']}")
