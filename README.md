# Document AI Assistant

A bilingual PDF question-answering application built around **retrieval-augmented generation (RAG)**, document structure extraction, and verifiable source references.

**Live demo:** https://emh-document-ai-assistant.streamlit.app/  
**Status:** public beta · CPU-first, open-weight inference

## Overview

Document AI Assistant accepts multiple digital or scanned PDFs, extracts their content, builds a semantic search index, and generates answers grounded in retrieved passages. Each answer can be checked against the original document through a page-linked PDF viewer.

The project focuses on practical document AI engineering: OCR integration, retrieval quality, source provenance, local model execution, and constrained-resource deployment.

## Implemented functionality

- **Document ingestion:** multiple PDFs, with document-specific or cross-document search.
- **Extraction:** Docling for document structure and tables; RapidOCR as Docling's configured OCR engine for scanned content, using ONNX Runtime.
- **Retrieval:** multilingual sentence-transformer embeddings combined with lexical overlap scoring.
- **Generation:** local Qwen3-1.7B Q4_K_M (GGUF) via `llama-cpp-python`, without visitor-provided API keys.
- **Evidence inspection:** page citations, integrated PDF preview, and highlighting where extraction coordinates are available.
- **Interface:** Streamlit UI in Spanish and English, with source navigation and timing information.
- **Resource constraints:** PDF size/page limits, bounded inference context, and synchronized access to the shared GGUF model.

Experimental optional modes for hosted generation and visual figure descriptions exist in the codebase but are **not part of the default public demo**.

## Architecture

```text
PDF upload
  ├─ PyMuPDF: page inspection
  └─ Docling: structure and table extraction
       └─ RapidOCR / ONNX Runtime (when OCR is required)
            ↓
Page-aware passages and available bounding boxes
            ↓
Sentence Transformers: multilingual embeddings
            ↓
Hybrid semantic + lexical retrieval
            ↓
Page / section context assembly
            ↓
Qwen3-1.7B GGUF via llama.cpp (CPU)
            ↓
Answer with source references + PDF viewer
```

### Extraction and OCR

Docling orchestrates the document-conversion pipeline. **RapidOCR is its selected OCR backend**, not a separate replacement for Docling. It is enabled when the PDF contains pages with insufficient selectable text. Docling also processes document structure and tables.

The application retains page identifiers and bounding boxes where supplied by the extraction pipeline. Coordinates can be missing or imprecise on some scans, so highlighting is only displayed when a source location is available.

### Retrieval and generation

Compact search passages are indexed with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Retrieved evidence is expanded to nearby page or section context before being sent to the local language model.

The default answer model is `ggml-org/Qwen3-1.7B-GGUF` (`Q4_K_M`) through llama.cpp. A bounded context reduces the risk of exceeding the model window, but long sections may still lose detail. The system presents candidate or cited source pages so users can inspect the underlying evidence.

## Run locally

Requires Python 3.11+ and sufficient RAM for OCR, embeddings, and GGUF inference.

```bash
git clone https://github.com/EMH01/document-ai-assistant.git
cd document-ai-assistant
python -m venv .venv
# Activate .venv (PowerShell: .venv\Scripts\Activate.ps1)
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The initial run downloads public model weights. `requirements.txt` includes a CPU wheel index for llama.cpp; installation requirements can vary across operating systems and Python versions.

Copy `.env.example` to `.env` to customize configuration. The default `AI_PROVIDER=gguf` uses local inference. `DOC_OCR_BACKEND=onnxruntime` selects the OCR runtime. Experimental provider and vision settings are optional.

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m compileall -q app.py src tests
```

Unit tests exercise retrieval, source metadata, OCR selection, academic-section boundaries, model-provider interfaces, and context limits. Real-document tests remain important because extraction quality depends on layout and scan quality.

## Public demo and limitations

**Demo:** https://emh-document-ai-assistant.streamlit.app/

The application runs on Streamlit Community Cloud with free-tier CPU resources. Scanned-document processing and generation can take a minute or more, especially on cold starts. Inference is serialized to protect shared model state, so simultaneous visitors may wait.

The beta accepts PDFs up to 15 MB and rejects files over 60 pages or password-protected documents. Complex tables, OCR artifacts, long document sections, and missing bounding boxes can affect answer quality or evidence highlighting.

**Privacy:** Uploaded content is processed on the application server. Do not upload confidential, medical, financial, or identifying documents to the public demo. Default GGUF generation does not send excerpts to a paid external inference provider.

## Possible Future Enhancements

- Asynchronous document processing and per-file progress tracking.
- Better table and scanned-PDF extraction evaluation.
- Latency and memory benchmarks for constrained deployment.
- More precise highlighting of OCR-derived evidence.
- Regression tests covering varied document layouts.

