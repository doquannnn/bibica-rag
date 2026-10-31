# Bibica RAG

PO extraction (VLM: Gemini API or local Qwen) and RAG over Bibica internal HR documents (salary/bonus regulation, labor rules, collective labor agreement),
plus a benchmark of RAG techniques (chunking, embeddings, reranking, LLM-wiki, LangGraph/LangChain/LlamaIndex/Haystack/DSPy)
using a local Qwen3.5 VLM served by llama.cpp.

## Setup
- Python 3.12 via `uv sync`
- Put the original `.doc` files in `RAG/` (not tracked: internal documents)
- `soffice --headless --convert-to docx --outdir data/docx RAG/*.doc`
- `uv run python -m bibica_rag.ingest.parse_docx` → `data/parsed/` (Markdown + `nodes.jsonl`)

Models (`models/`) and llama.cpp binaries (`tools/`) are downloaded locally and not tracked.

## Web UI
`uv run python app.py` (http://127.0.0.1:7860) — tabs **Hỏi đáp** (RAG pipelines from the benchmark),
**Trích xuất PO** and **Đọc ảnh (VLM)**. The local VLM is started with `scripts/serve.sh 4B`.

## PO extraction (`bibica_rag/po/`)
Retailer purchase order PDF → exactly the fields Bibica needs for that retailer (from the Excel in `PO Extraction/`),
plus the DMS customer code. Details, evaluation method and current results: `docs/PO_EXTRACTION.md`.

```bash
scripts/serve.sh 4B              # local Qwen3.5-4B VLM on :8080 (default); or BIBICA_PO_PROVIDER=gemini GEMINI_API_KEY=...
uv run python -m bibica_rag.po.extract "PO Extraction/" --out results/po
uv run python -m bibica_rag.po.evaluate score --pred results/po --gt data/po_gt     # -> results/po_eval/report.md
uv run pytest tests/                                     # unit tests + model-free checks on the real samples
BIBICA_LIVE=1 uv run pytest tests/test_po_live.py -s     # accuracy gate with the real VLM
```
