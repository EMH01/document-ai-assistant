import os

import streamlit as st
from openai import OpenAI

from document_ai.config import Settings
from document_ai.llm import answer_question
from document_ai.pdf import extract_chunks
from document_ai.retrieval import embed_chunks, retrieve

st.set_page_config(page_title="Document AI Assistant", page_icon="📄", layout="wide")
settings = Settings()

st.title("📄 Document AI Assistant")
st.caption("Multimodal PDF ingestion + semantic retrieval + grounded answers")

if not os.getenv("OPENAI_API_KEY"):
    st.info("Set OPENAI_API_KEY in your environment or .env file to run the demo.")
    st.stop()

client = OpenAI()

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    include_images = st.toggle(
        "Describe embedded images",
        value=settings.enable_image_captions,
        help=(
            "Uses the configured chat model during ingestion to convert images "
            "into searchable captions."
        ),
    )
    process = st.button("Process document", type="primary", disabled=uploaded is None)

    if st.session_state.chunks:
        st.metric("Indexed chunks", len(st.session_state.chunks))
        if st.button("Clear document"):
            st.session_state.chunks = []
            st.session_state.messages = []
            st.rerun()

if process and uploaded is not None:
    with st.status("Indexing document...", expanded=True) as status:
        st.write("Extracting text and document structure")
        chunks = extract_chunks(
            uploaded.getvalue(),
            source=uploaded.name,
            client=client,
            vision_model=settings.chat_model,
            include_image_captions=include_images,
        )
        if not chunks:
            status.update(label="No readable content found", state="error")
            st.stop()
        st.write(f"Creating embeddings for {len(chunks)} chunks")
        embed_chunks(client, chunks, settings.embedding_model)
        st.session_state.chunks = chunks
        st.session_state.messages = []
        status.update(label="Document indexed", state="complete")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask a question about the document",
    disabled=not bool(st.session_state.chunks),
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context..."):
            retrieved = retrieve(
                client,
                st.session_state.chunks,
                question,
                settings.embedding_model,
                top_k=settings.top_k,
            )
            answer = answer_question(client, question, retrieved, settings.chat_model)
        st.markdown(answer)
        with st.expander("Retrieved context"):
            for index, (chunk, score) in enumerate(retrieved, start=1):
                st.markdown(f"**[{index}] {chunk.citation} — {score:.3f}**")
                st.write(chunk.text)

    st.session_state.messages.append({"role": "assistant", "content": answer})
