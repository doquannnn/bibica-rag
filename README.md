# Bibica RAG

Local RAG over Bibica internal HR documents (salary/bonus regulation, labor rules, collective labor agreement),
plus a benchmark of RAG techniques (chunking, embeddings, reranking, LLM-wiki, LangGraph/LangChain/LlamaIndex/Haystack/DSPy)
using a local Qwen3.5 VLM served by llama.cpp.

## Setup
- Python 3.12 via `uv sync`
- Put the original `.doc` files in `RAG/` (not tracked: internal documents)
- `soffice --headless --convert-to docx --outdir data/docx RAG/*.doc`
- `uv run python -m bibica_rag.ingest.parse_docx` → `data/parsed/` (Markdown + `nodes.jsonl`)

Models (`models/`) and llama.cpp binaries (`tools/`) are downloaded locally and not tracked.

## PO extraction (cloud)
See `docs/PO_EXTRACTION.md` for the brief and `PO Extraction/` for the sample purchase orders and the store/DMS mapping workbook.
