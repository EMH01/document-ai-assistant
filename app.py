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

if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "language" not in st.session_state:
    st.session_state.language = "es"

with st.sidebar:
    st.selectbox("Idioma / Language", ["Español", "English"], key="language_choice", index=0 if st.session_state.language == "es" else 1)
st.session_state.language = "es" if st.session_state.language_choice == "Español" else "en"
LANG = {
    "es": {"title": "Consulta tus documentos con respuestas basadas en fuentes.", "document": "Documento", "upload": "Sube un PDF", "loading": "Extrayendo páginas y preparando la búsqueda…", "ready": "Documento preparado", "fragments": "fragmentos consultables", "clear": "Quitar documento", "ask": "Pregunta sobre tu documento", "searching": "Buscando fuentes y preparando la respuesta…", "sources": "Fuentes consultadas", "pages": "página", "score": "puntuación de recuperación", "info": "Los archivos se procesan en el servidor. Evita subir documentos sensibles a demos públicas.", "auto": "El documento se procesará automáticamente al cargarlo.", "indexed": "Documento listo para consultar", "ocr": "No se ha podido extraer texto. Verifica que Docling y su motor OCR estén instalados y disponibles en el servidor.", "oversize": "El PDF supera el límite de 15 MB."},
    "en": {"title": "Ask your documents questions with source-grounded answers.", "document": "Document", "upload": "Upload a PDF", "loading": "Extracting pages and building the search index…", "ready": "Document ready", "fragments": "searchable excerpts", "clear": "Remove document", "ask": "Ask a question about your document", "searching": "Finding sources and preparing an answer…", "sources": "Retrieved sources", "pages": "page", "score": "retrieval score", "info": "Files are processed on the host. Avoid uploading sensitive documents to public demos.", "auto": "The document is processed automatically after upload.", "indexed": "Document ready for questions", "ocr": "No readable text extracted. Verify Docling and the OCR engine are installed and available on the host.", "oversize": "The PDF exceeds the 15 MB limit."}
}
t = LANG[st.session_state.language]
settings = Settings()
st.title("📄 Document AI Assistant")
st.caption(t["title"])

with st.sidebar:
    use_openai = settings.provider == "openai"
    st.header(t["document"])
    uploaded = st.file_uploader(t["upload"], type=["pdf"], max_upload_size=15, key=f"uploaded_{st.session_state.upload_key}")
    st.caption(t["auto"])
    if st.session_state.get("chunks"):
        st.caption(f"{t['ready']} · {len(st.session_state.chunks)} {t['fragments']}")
        if st.button(t["clear"]):
            st.session_state.chunks = []
            st.session_state.messages = []
            st.session_state.provider_id = None
            st.session_state.document_digest = None
            st.session_state.upload_key += 1
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

import hashlib

uploaded_digest = hashlib.sha256(uploaded.getvalue()).hexdigest() if uploaded else None
if uploaded is not None and uploaded_digest != st.session_state.get("document_digest"):
    if uploaded.size > 15 * 1024 * 1024:
        st.error(t["oversize"])
        st.stop()
    try:
        embeddings, _, client = get_clients()
        with st.spinner(t["loading"]):
            chunks = extract_document(uploaded.getvalue(), source=uploaded.name)
            if not chunks:
                st.error(t["ocr"])
                st.stop()
            index_chunks(embeddings, chunks)
            st.session_state.chunks = chunks
            st.session_state.messages = []
            st.session_state.provider_id = provider_id
            st.session_state.document_digest = uploaded_digest
        st.success(f"{t['indexed']} · {len(chunks)} {t['fragments']}.")
    except Exception as exc:
        st.error(f"Document processing failed: {exc}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(t["ask"], disabled=not bool(st.session_state.chunks))
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            embeddings, answerer, _ = get_clients()
            with st.spinner(t["searching"]):
                query_vec = embeddings.encode([question])[0]
                ranked = hybrid_rank_chunks(st.session_state.chunks, np.asarray(query_vec), question, settings.top_k)
                response = answer_from_chunks(answerer, question, ranked)
            st.markdown(response)
            with st.expander(t["sources"]):
                for idx, (chunk, score) in enumerate(ranked, 1):
                    st.markdown(f"**[{idx}] {chunk.citation} — {t['score']} {score:.3f}**")
                    st.write(chunk.text)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as exc:
            st.error(f"Answer generation failed: {exc}")

st.caption(t["info"])
