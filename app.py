import os

import numpy as np
import streamlit as st

from document_ai.config import Settings
from document_ai.docling_extract import extract_document
from document_ai.providers import (
    LocalAnswer, LocalEmbeddings, OpenAIAnswer, OpenAIEmbeddings,
    answer_from_chunks, index_chunks,
)
from document_ai.retrieval import hybrid_rank_chunks

st.set_page_config(page_title="Document AI Assistant", page_icon="📄", layout="wide")
settings = Settings()
st.title("📄 Document AI Assistant")
st.caption("Explore your PDF with source-grounded answers. Open-weight models by default.")

with st.sidebar:
    use_openai = settings.provider == "openai"
    st.header("Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], max_upload_size=15)
    process = st.button("Process document", type="primary", disabled=uploaded is None)
    if st.session_state.get("chunks"):
        st.caption(f"Documento preparado · {len(st.session_state.chunks)} fragmentos consultables")
        if st.button("Clear document"):
            st.session_state.chunks = []
            st.session_state.messages = []
            st.session_state.provider_id = None
            st.rerun()

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "provider_id" not in st.session_state:
    st.session_state.provider_id = None

provider_id = "openai" if use_openai else "open"
if st.session_state.provider_id is not None and st.session_state.provider_id != provider_id:
    st.warning("Provider changed: process the PDF again to rebuild its embeddings.")
    st.session_state.chunks = []
    st.session_state.messages = []
    st.session_state.provider_id = None

def get_clients():
    if not use_openai:
        return LocalEmbeddings(settings.local_embedding_model), LocalAnswer(settings.local_chat_model), None
    from openai import OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Configure OPENAI_API_KEY on the host for OpenAI mode.")
    client = OpenAI()
    return OpenAIEmbeddings(client, settings.embedding_model), OpenAIAnswer(client, settings.chat_model), client

if process and uploaded is not None:
    if uploaded.size > 15 * 1024 * 1024:
        st.error("Maximum PDF size: 15 MB.")
        st.stop()
    try:
        embeddings, _, client = get_clients()
        with st.spinner("Extracting pages and building the semantic index..."):
            chunks = extract_document(uploaded.getvalue(), source=uploaded.name)
            if not chunks:
                st.error("No readable text found. Scanned PDFs need OCR, not yet implemented.")
                st.stop()
            index_chunks(embeddings, chunks)
            st.session_state.chunks = chunks
            st.session_state.messages = []
            st.session_state.provider_id = provider_id
        st.success(f"Indexed {len(chunks)} excerpts.")
    except Exception as exc:
        st.error(f"Document processing failed: {exc}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your PDF", disabled=not bool(st.session_state.chunks))
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            embeddings, answerer, _ = get_clients()
            with st.spinner("Searching the PDF and generating an answer..."):
                query_vec = embeddings.encode([question])[0]
                ranked = hybrid_rank_chunks(st.session_state.chunks, np.asarray(query_vec), question, settings.top_k)
                response = answer_from_chunks(answerer, question, ranked)
            st.markdown(response)
            with st.expander("Retrieved source excerpts"):
                for idx, (chunk, score) in enumerate(ranked, 1):
                    st.markdown(f"**[{idx}] {chunk.citation} — retrieval score {score:.3f}**")
                    st.write(chunk.text)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as exc:
            st.error(f"Answer generation failed: {exc}")

st.caption("Models run on the host. Uploaded content may be processed by the hosting server; avoid sensitive PDFs in public demos.")
