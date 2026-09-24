import streamlit as st

from retrieval.pipeline.query import ask
from common.catalog import list_books


st.set_page_config(page_title="Ask — Kannada RAG", page_icon="🔎")

st.title("🔎 Ask a question")
st.caption("Ask a question in Kannada. Answers are grounded in the indexed books below.")

books = list_books()

with st.sidebar:
    st.header("Settings")

    book_options = {"All books": None}
    book_options.update({f'{b["title"]} ({b["book_id"]})': b["book_id"] for b in books})

    selected_label = st.selectbox("Search within", list(book_options.keys()))
    selected_book_id = book_options[selected_label]

    st.caption(
        "Retrieval depth and whether the self-critique loop runs are decided "
        "automatically per question by the router."
    )

    st.divider()
    st.caption("Indexed books:")
    for b in books:
        st.caption(f"- {b['title']} ({b['book_id']})")

question = st.text_input("Your question (ಕನ್ನಡದಲ್ಲಿ ಪ್ರಶ್ನೆ ಕೇಳಿ)")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    where = {"book_id": selected_book_id} if selected_book_id else None

    with st.spinner("Retrieving and generating answer..."):
        result = ask(question, where=where)

    badge = f"Route: **{result['route']}**"
    if result["cache_hit"]:
        badge += " · cached"
    if result.get("iteration_count"):
        badge += f" · {result['iteration_count']} agent iteration(s)"
    st.caption(badge)

    st.subheader("Answer")

    if result["error"]:
        st.warning(result["error"])
    elif result["cache_hit"] or result["answer_stream"] is None:
        st.write(result["answer"])
    else:
        st.write_stream(result["answer_stream"])
        stream_error = result.get("stream_error", {}).get("error")
        if stream_error:
            st.warning(stream_error)

    st.subheader("Sources")

    for rank, chunk in enumerate(result["reranked"], start=1):
        with st.expander(
            f"{rank}. {chunk['title']} — page {chunk['page_start']}-{chunk['page_end']} "
            f"(score {chunk.get('rerank_score', chunk.get('score', 0)):.3f})"
        ):
            st.caption(f"Author: {chunk['author']} · Chunk: {chunk['chunk_id']}")
            st.write(chunk["text"])

elif ask_clicked:
    st.warning("Please enter a question.")
