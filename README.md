<div align="center">

# 📄 Document AI Assistant

### Ask your PDFs. Verify the evidence.

**A bilingual, source-grounded document assistant powered by local open-weight AI, OCR, and semantic retrieval.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20in%20Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://emh-document-ai-assistant.streamlit.app/)
[![Source Code](https://img.shields.io/badge/Source-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/EMH01/document-ai-assistant)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/Status-Public%20Beta-64748b)

**[Try the live app →](https://emh-document-ai-assistant.streamlit.app/)** · **[Explore the code →](https://github.com/EMH01/document-ai-assistant)**

</div>

---

## The experience

Upload one or more PDFs, ask questions in **English or Spanish**, and inspect the pages behind the answer. The application is designed around **verifiable evidence**, not just plausible responses.

| Capability | What it does |
| --- | --- |
| **Digital & scanned PDFs** | Extracts document text with Docling and RapidOCR for pages requiring OCR. |
| **Multi-document Q&A** | Searches across uploaded documents or within one selected PDF. |
| **Local AI generation** | Runs a quantized **Qwen3-1.7B GGUF** model through `llama.cpp` on the host CPU. No visitor API key required. |
| **Semantic + lexical retrieval** | Uses multilingual sentence-transformer embeddings and hybrid ranking to find supporting passages. |
| **Inspect the evidence** | Opens cited or retrieved PDF pages in an integrated viewer, highlighting source regions when valid coordinates are available. |
| **Bilingual interface** | Spanish / English interface and question answering. |

> **Live beta:** OCR and model inference run on CPU and may take over a minute for scanned PDFs or answers, especially on a cold start. The demo is for exploration, not a high-throughput document service.

## Try it

1. Open **[Document AI Assistant on Streamlit](https://emh-document-ai-assistant.streamlit.app/)**.
2. Upload a **non-sensitive PDF** using the sidebar (up to 15 MB; documents over 60 pages and encrypted PDFs are not supported in the public beta).
3. Wait for extraction and indexing, then ask a question such as:
   - *What are the key conclusions of this document?*
   - *What obligations does the agreement describe?*
   - *¿Qué se establece sobre los plazos?*
4. Open a source to inspect the corresponding PDF page.

**Privacy:** Files are processed on the app server, not solely in the browser. Do **not** upload confidential, medical, financial, or identifying documents to the public demo. Optional hosted-provider modes are separate from the default local generation.

## How it works

```text
PDF upload
   │
   ├── PyMuPDF inspection + Docling structure extraction
   │                             └── RapidOCR (for scanned pages)
   ▼
Page-aware excerpts + available bounding boxes
   ▼
Multilingual Sentence Transformers embeddings
   ▼
Hybrid semantic / lexical search
   ▼
Page / section context assembly
   ▼
Qwen3-1.7B (GGUF · llama.cpp · CPU)
   ▼
Natural-language answer + source links + PDF viewer
```

### Design decisions

- **Explainable answers:** retrieval preserves source/page metadata; PDF highlights appear only when coordinates can be verified.
- **Open by default:** local quantized generation rather than requiring a paid inference API.
- **Practical CPU deployment:** ONNX Runtime OCR and bounded context for a limited-memory host.
- **Honest degradation:** scanned text, complex tables, model output, and OCR coordinates can be imperfect; answers should be checked against the original page.
- **Document safety limits:** this public beta rejects encrypted documents and PDFs over 60 pages to avoid unbounded work. Concurrent GGUF inference is serialized.

## Run locally

Python **3.11+** is required. The GGUF model and embedding weights are downloaded on first use; sufficient RAM and an internet connection for the initial download are needed.

```bash
git clone https://github.com/EMH01/document-ai-assistant.git
cd document-ai-assistant
python -m venv .venv
# Activate your virtual environment.
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`. On macOS/Linux, use `source .venv/bin/activate`.

The requirements file includes the CPU wheel index for `llama-cpp-python`. Platform-specific builds may require additional compilation tools.

### Configuration

Copy `.env.example` to `.env` and modify settings as needed. By default:

| Setting | Default / role |
| --- | --- |
| `AI_PROVIDER` | `gguf` — local generation |
| `LOCAL_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `DOC_OCR_BACKEND` | `onnxruntime` |
| `VISUAL_CAPTIONS` | `false` — experimental visual descriptions disabled |
| `TOP_K` | `3` (unless overridden) |

Alternative provider modes (`hf`, `open`, and `openai`) exist for experimentation, but the public deployment uses the local GGUF path. Optional remote providers can require credentials and may transmit excerpts outside the host.

## Quality checks

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m compileall -q app.py src tests
```

Tests cover source rendering, retrieval, academic-section continuity, provider behavior, OCR selection, and basic deployment checks. Passing unit tests does not guarantee that every PDF layout will be parsed perfectly.

## Current limitations & roadmap

- **Latency:** CPU-based OCR and LLM generation are slower than GPU-hosted alternatives.
- **Concurrency:** generation is serialized for model safety; queues may form under simultaneous use.
- **Complex PDFs:** tables, unusual layouts, and OCR artifacts can affect exact counts and source highlights.
- **Planned improvements:** background upload queue, better per-document status, stronger table-structure evaluation, performance benchmarks, and improved scanned-PDF highlights.

---

<div align="center">

**Built as an applied AI engineering portfolio project.**

[Launch live demo](https://emh-document-ai-assistant.streamlit.app/) · [GitHub profile](https://github.com/EMH01)

</div>
