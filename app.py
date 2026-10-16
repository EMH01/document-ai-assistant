"""Streamlit document workspace: multilingual, multi-document RAG, source preview."""
import hashlib
import re
import os
import time


import pymupdf as fitz
import numpy as np
import streamlit as st

from document_ai.config import Settings
from document_ai.docling_extract import extract_document
from document_ai.hierarchical import build_search_index, expand_with_page_neighbors, page_context_blocks
from document_ai.visual import pdf_visual_regions, describe_visuals
from document_ai.providers import (
    LocalAnswer, LocalEmbeddings, HuggingFaceAnswer, FallbackAnswer, OpenAIAnswer, OpenAIEmbeddings,
    answer_from_chunks, index_chunks,
)
from document_ai.retrieval import hybrid_rank_chunks

st.set_page_config(page_title="Document AI Assistant", page_icon="📄", layout="wide")

TEXT = {
    "es": {
        "tagline": "Pregunta sobre uno o varios PDF y consulta las fuentes originales.",
        "files": "Documentos", "upload": "Añadir archivos PDF", "scope": "Consultar",
        "all": "Todos los documentos", "ready": "Listo", "pages": "página",
        "fragments": "fragmentos", "clear": "Vaciar documentos", "ask": "Pregunta sobre tus documentos",
        "extract": "Extrayendo texto, tablas y OCR…", "embed": "Preparando búsqueda semántica…",
        "answer": "Buscando y generando una respuesta…", "sources": "Fuentes",
        "preview": "Visor de documento", "choose": "Selecciona una fuente para verla",
        "no_coords": "No hay coordenadas verificables para este fragmento: se muestra la página sin resaltar.",
        "ocr_error": "No se pudo extraer texto. Revisa la configuración OCR del servidor.",
        "privacy": "Los documentos se procesan en el servidor. Evita subir datos sensibles a demos públicas.",
        "missing": "Docling no está instalado; el OCR no estará disponible.",
        "no_source": "No hay archivos procesados todavía.",
        "failed": "Error durante el procesamiento",
        "processing": "Procesando", "score": "puntuación", "total": "Archivos preparados",
        "show_viewer": "Mostrar visor", "hide_viewer": "Ocultar visor",
        "extract_time": "Extracción/OCR", "index_time": "Indexación", "total_time": "Tiempo total",
        "search_time": "Búsqueda", "generation_time": "Generación",
        "candidate_sources": "Fragmentos recuperados (no necesariamente citados por la respuesta)",
        "cited_source": "Citada", "candidate_source": "Candidata",
        "interaction_note": "Si envías una pregunta, espera su respuesta antes de añadir archivos: Streamlit puede reiniciar la ejecución activa.",
    },
    "en": {
        "tagline": "Ask questions about one or more PDFs and inspect the original sources.",
        "files": "Documents", "upload": "Add PDF files", "scope": "Search in",
        "all": "All documents", "ready": "Ready", "pages": "page",
        "fragments": "excerpts", "clear": "Clear documents", "ask": "Ask about your documents",
        "extract": "Extracting text, tables and OCR…", "embed": "Building semantic search…",
        "answer": "Searching and generating an answer…", "sources": "Sources",
        "preview": "Document viewer", "choose": "Select a source to view it",
        "no_coords": "Exact coordinates unavailable for this excerpt; showing the page without highlighting.",
        "ocr_error": "No text extracted. Check server OCR configuration.",
        "privacy": "Documents are processed on the host. Avoid sensitive files in public demos.",
        "missing": "Docling is not installed; OCR is unavailable.",
        "no_source": "No documents processed yet.",
        "failed": "Processing failed", "processing": "Processing", "score": "score",
        "total": "Ready documents", "show_viewer": "Show viewer", "hide_viewer": "Hide viewer",
        "extract_time": "Extraction/OCR", "index_time": "Indexing", "total_time": "Total time",
        "search_time": "Retrieval", "generation_time": "Generation",
        "candidate_sources": "Retrieved excerpts (not necessarily cited by the answer)",
        "cited_source": "Cited", "candidate_source": "Candidate",
        "interaction_note": "Wait for the current answer before uploading another file: Streamlit may rerun and interrupt active generation.",
    },
}

for key, initial in {
    "files": {}, "messages": [], "upload_key": 0, "language": "es",
    "selected_source": None, "provider_id": None, "viewer_open": False,
    "processed_cache": {}, "hf_failed": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = initial

settings = Settings()
try:
    streamlit_provider = st.secrets.get("AI_PROVIDER")
except (FileNotFoundError, KeyError):
    streamlit_provider = None
provider_id = os.getenv("AI_PROVIDER") or streamlit_provider or settings.provider
if st.session_state.provider_id not in (None, provider_id):
    st.session_state.files = {}
    st.session_state.processed_cache = {}
    st.session_state.messages = []
st.session_state.provider_id = provider_id

def get_clients():
    if provider_id == "hf":
        token = os.getenv("HF_TOKEN")
        if not token:
            try:
                token = st.secrets.get("HF_TOKEN")
            except (FileNotFoundError, KeyError):
                token = None
        local = LocalAnswer(settings.local_chat_model)
        if not token or st.session_state.hf_failed:
            st.session_state.hf_failed = True
            return LocalEmbeddings(settings.local_embedding_model), local
        return LocalEmbeddings(settings.local_embedding_model), FallbackAnswer(
            HuggingFaceAnswer(token, settings.hf_chat_model, timeout=15), local
        )
    if provider_id == "open":
        return LocalEmbeddings(settings.local_embedding_model), LocalAnswer(settings.local_chat_model)
    if provider_id != "openai":
        raise ValueError("AI_PROVIDER must be open, hf or openai")
    from openai import OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be configured on the host.")
    client = OpenAI()
    return OpenAIEmbeddings(client, settings.embedding_model), OpenAIAnswer(client, settings.chat_model)

with st.sidebar:
    language = st.selectbox(
        "Idioma / Language", ["Español", "English"],
        index=0 if st.session_state.language == "es" else 1,
    )
    st.session_state.language = "es" if language == "Español" else "en"
    t = TEXT[st.session_state.language]
    st.header(t["files"])
    st.caption(t["interaction_note"])
    uploaded = st.file_uploader(
        t["upload"], type=["pdf"], accept_multiple_files=True,
        max_upload_size=15, key=f"upload_{st.session_state.upload_key}",
    )
    if st.button(t["clear"], disabled=not st.session_state.files):
        st.session_state.files = {}
        st.session_state.processed_cache = {}
        st.session_state.messages = []
        st.session_state.selected_source = None
        st.session_state.upload_key += 1
        st.rerun()

# Separate filename from identity: duplicate names can contain different PDFs.
uploads = {}
for file in uploaded or []:
    raw = file.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    uploads[digest] = (file.name, raw)
if uploaded is not None:
    for digest in list(st.session_state.files):
        if digest not in uploads:
            del st.session_state.files[digest]
    for digest, (name, raw) in uploads.items():
        if digest in st.session_state.files:
            continue
        cached = st.session_state.processed_cache.get((provider_id, digest))
        if cached is not None:
            st.session_state.files[digest] = {**cached, "name": name, "bytes": raw}
            continue
        try:
            embeddings, _ = get_clients()
            started = time.perf_counter()
            with st.spinner(f"{t['processing']}: {name} — {t['extract']}"):
                chunks = extract_document(raw, source=name, require_docling=True)
                visual_regions = pdf_visual_regions(raw)
                if os.getenv("VISUAL_CAPTIONS", "false").lower() == "true":
                    chunks.extend(describe_visuals(raw, source=name, max_regions=2))
            after_extract = time.perf_counter()
            with st.spinner(f"{t['processing']}: {name} — {t['embed']}"):
                search_chunks, parent_map = build_search_index(chunks)
                index_chunks(embeddings, search_chunks)
            after_index = time.perf_counter()
            timings = {"extract": after_extract - started, "index": after_index - after_extract, "total": after_index - started}
            prepared = {"name": name, "bytes": raw, "chunks": chunks, "search_chunks": search_chunks, "parent_map": parent_map, "timings": timings, "visual_regions": len(visual_regions)}
            st.session_state.files[digest] = prepared
            # Private to this Streamlit session; never cache visitors' PDFs globally.
            st.session_state.processed_cache[(provider_id, digest)] = prepared
        except Exception as exc:
            st.error(f"{t['failed']}: {name} — {exc}")

all_files = st.session_state.files
names = {key: f"{value['name']} ({key[:6]})" for key, value in all_files.items()}
with st.sidebar:
    options = ["__all__", *names]
    scope = st.selectbox(t["scope"], options, format_func=lambda key: t["all"] if key == "__all__" else names[key])
    st.caption(f"{t['total']}: {len(all_files)}")
    if all_files:
        st.caption(f"{sum(len(v['chunks']) for v in all_files.values())} {t['fragments']}")

st.title("📄 Document AI Assistant")
st.caption(t["tagline"])
viewer_label = t["hide_viewer"] if st.session_state.viewer_open else t["show_viewer"]
if st.button(("◧ " if st.session_state.viewer_open else "▣ ") + viewer_label):
    st.session_state.viewer_open = not st.session_state.viewer_open
    st.rerun()
if st.session_state.viewer_open:
    main, right = st.columns([2.25, 1], gap="medium")
else:
    main = st.container()
    right = None

def draw_pdf(raw: bytes, page_number: int, bbox=None):
    with fitz.open(stream=raw, filetype="pdf") as document:
        page_number = max(1, min(page_number, len(document)))
        page = document[page_number - 1]
        scale = min(1.4, 1100 / max(page.rect.width, 1))
        if bbox is not None:
            try:
                rect = fitz.Rect(*bbox) & page.rect
                if not rect.is_empty and rect.get_area() > 0:
                    page.draw_rect(rect, color=(0.82, 0.55, 0.0), fill=(1.0, 0.9, 0.45), fill_opacity=0.30, width=1.5, overlay=True)
                else:
                    bbox = None
            except (ValueError, TypeError):
                bbox = None
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        image_data = pix.tobytes("png")
    st.image(image_data, width="stretch")
    # Highlight only from verified document provenance, never guessed from model output.
    if bbox is None:
        st.caption(t["no_coords"])

with main:
    st.subheader(t["files"])
    if all_files:
        for entry in all_files.values():
            times = entry.get("timings", {})
            duration = f" · {t['extract_time']}: {times['extract']:.1f}s · {t['index_time']}: {times['index']:.1f}s · {t['total_time']}: {times['total']:.1f}s" if times else ""
            st.caption(f"✓ {entry['name']} · {len(entry['chunks'])} {t['fragments']} · Visual: {entry.get('visual_regions', 0)}{duration}")
    else:
        st.info(t["no_source"])
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("timings"):
                timings = message["timings"]
                st.caption(f"{t['search_time']}: {timings['retrieval']:.1f}s · {t['generation_time']}: {timings['generation']:.1f}s · {t['total_time']}: {timings['total']:.1f}s")
            if message["role"] == "assistant" and message.get("ranked"):
                with st.expander(t["sources"] + " · " + t["candidate_sources"]):
                    for j, chunk, score, file_id in message["ranked"]:
                        status = t["cited_source"] if j in message.get("cited_numbers", set()) else t["candidate_source"]
                        if st.button(
                            f"[{j}] {chunk.source} · {t['pages']} {chunk.page} · {status} · {t['score']} {score:.2f}",
                            key=f"source_{idx}_{j}",
                        ):
                            st.session_state.selected_source = (file_id, chunk.page, chunk.bbox)
                            st.session_state.viewer_open = True
                            st.rerun()
                        st.caption(chunk.text[:220])
# Keep the chat composer at the bottom of the page, independently of viewer width.
question = st.chat_input(t["ask"], disabled=not all_files)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            embeddings, answerer = get_clients()
            with st.spinner(t["answer"]):
                answer_started = time.perf_counter()
                vector = np.asarray(embeddings.encode([question])[0])
                selected = all_files if scope == "__all__" else {scope: all_files[scope]}
                children = [child for entry in selected.values() for child in entry["search_chunks"]]
                parents = {k: v for entry in selected.values() for k, v in entry["parent_map"].items()}
                ranked_children = hybrid_rank_chunks(children, vector, question, settings.top_k)
                original_chunks = [chunk for entry in selected.values() for chunk in entry["chunks"]]
                matched = expand_with_page_neighbors(
                    ranked_children, parents, original_chunks, max_excerpts=40
                )
                if matched:
                    focused = matched[0][0]
                    page_chunks = page_context_blocks(
                        original_chunks, focused.source,
                        {focused.page, focused.page + 1}
                    )
                    ranked = [(chunk, matched[0][1]) for chunk in page_chunks]
                else:
                    ranked = []
                retrieval_finished = time.perf_counter()
                answer = answer_from_chunks(answerer, question, ranked)
                if isinstance(answerer, FallbackAnswer) and answerer.fell_back:
                    st.session_state.hf_failed = True
                generation_finished = time.perf_counter()
                raw_citations = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
                cited_numbers = {
                    n for n in raw_citations
                    if 1 <= int(n) <= len(ranked)
                }
                if any(n > len(ranked) or n < 1 for n in raw_citations):
                    answer = re.sub(r"\[(\d+)\]", lambda match: match.group(0) if 1 <= int(match.group(1)) <= len(ranked) else "", answer).strip()
                    answer += ("\n\n*Las referencias no verificables se han omitido.*" if st.session_state.language == "es" else "\n\n*Unverifiable references were omitted.*")
                result = []
                for number, (chunk, score) in enumerate(ranked, 1):
                    file_id = next(key for key, entry in selected.items() if entry["name"] == chunk.source)
                    result.append((number, chunk, score, file_id))
                st.session_state.messages.append({"role": "assistant", "content": answer, "ranked": result, "cited_numbers": cited_numbers, "timings": {"retrieval": retrieval_finished - answer_started, "generation": generation_finished - retrieval_finished, "total": generation_finished - answer_started}})
            st.rerun()
        except Exception as exc:
            st.error(f"{t['failed']}: {exc}")

if right is not None:
    with right:
        st.subheader(t["preview"])
        selected_source = st.session_state.selected_source
        if selected_source and selected_source[0] in all_files:
            file_id, page_number, bbox = selected_source
        elif all_files:
            file_id = next(iter(all_files))
            page_number, bbox = 1, None
        else:
            file_id = None
        if file_id:
            entry = all_files[file_id]
            with fitz.open(stream=entry["bytes"], filetype="pdf") as document:
                total_pages = len(document)
            st.caption(entry["name"])
            page_number = st.number_input(
                t["pages"], min_value=1, max_value=total_pages,
                value=max(1, min(page_number, total_pages)),
                key=f"page_{file_id}_{page_number}",
            )
            draw_pdf(entry["bytes"], page_number, bbox if selected_source and selected_source[0] == file_id and selected_source[1] == page_number else None)
        else:
            st.info(t["choose"])
    
st.divider()
st.caption(t["privacy"])
