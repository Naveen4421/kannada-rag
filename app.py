import streamlit as st

from src.pipeline.catalog import list_books


st.set_page_config(page_title="Kannada RAG", page_icon="📚")

st.title("📚 Kannada RAG")
st.caption("Two separate pipelines, one app.")

st.markdown(
    "- **📥 Ingest Data** (see sidebar) — the *dataset pipeline*. Upload a `.docx`/`.txt` "
    "book file; it gets loaded, chunked, embedded, and indexed into the shared vector search collection.\n"
    "- **🔎 Ask Questions** (see sidebar) — the *query pipeline*. Ask a question in Kannada; it "
    "gets embedded, routed, searched, reranked, and answered by the LLM."
)

books = list_books()
st.subheader(f"Currently indexed: {len(books)} book(s)")
for b in books:
    st.caption(f"- {b['title']} ({b['book_id']}) — {b['author']}")
