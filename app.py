import json
from pathlib import Path

import streamlit as st

from src.pipeline.query import answer_question


st.set_page_config(page_title="Kannada RAG", page_icon="📚")


@st.cache_data
def list_books():
    books = []

    for path in sorted(Path("data/processed").glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            doc = json.load(f)

        if "pages" in doc and doc["pages"] and "paragraphs" in doc["pages"][0]:
            books.append({"book_id": doc["book_id"], "title": doc["title"]})

    return books


st.title("Kannada RAG")
st.caption("Ask a question in Kannada. Answers are grounded in the indexed books below.")

books = list_books()

with st.sidebar:
    st.header("Settings")

    book_options = {"All books": None}
    book_options.update({f'{b["title"]} ({b["book_id"]})': b["book_id"] for b in books})

    selected_label = st.selectbox("Search within", list(book_options.keys()))
    selected_book_id = book_options[selected_label]

    top_k = st.slider("Chunks retrieved (top_k)", min_value=5, max_value=30, value=10)
    top_n = st.slider("Chunks sent to LLM (top_n)", min_value=1, max_value=10, value=5)

    st.divider()
    st.caption("Indexed books:")
    for b in books:
        st.caption(f"- {b['title']} ({b['book_id']})")

question = st.text_input("Your question (ಕನ್ನಡದಲ್ಲಿ ಪ್ರಶ್ನೆ ಕೇಳಿ)")
ask = st.button("Ask", type="primary")

if ask and question.strip():
    where = {"book_id": selected_book_id} if selected_book_id else None

    with st.spinner("Retrieving and generating answer..."):
        result = answer_question(question, top_k=top_k, top_n=top_n, where=where)

    st.subheader("Answer")

    if result["error"]:
        st.warning(result["error"])
    else:
        st.write(result["answer"])

    st.subheader("Sources")

    for rank, chunk in enumerate(result["reranked"], start=1):
        with st.expander(
            f"{rank}. {chunk['title']} — page {chunk['page_start']}-{chunk['page_end']} "
            f"(score {chunk['score']:.3f})"
        ):
            st.caption(f"Author: {chunk['author']} · Chunk: {chunk['chunk_id']}")
            st.write(chunk["text"])

elif ask:
    st.warning("Please enter a question.")
