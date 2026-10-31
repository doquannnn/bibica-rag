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
`uv run python app.py` (http://127.0.0.1:7860) — tabs **Hỏi đáp** (RAG pipelines from the benchmark, chosen under
**Cấu hình**) and **Trích xuất PO**. The local VLM is started with `scripts/serve.sh 4B` (:8080); start
`scripts/serve.sh 9B 8082` as well to switch to Qwen3.5-9B with the **Model** dropdown at the top (used by both tabs;
PO only with provider `local`, set under **Cấu hình**). Other endpoints: `LLM_ENDPOINTS="name=url,..."`.

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

## Stage C: Qwen3.5-9B
`scripts/bench_9b.sh` re-runs the Stage-B top pipelines (CAG, DSPy, naive-best-s2b, naive-best) with the 9B model,
unattended and resumable, and packs everything into `results/stageC_9B.tar.gz` for judging.
