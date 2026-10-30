# Bibica RAG

Local PO extraction (Qwen VLM) and RAG over Bibica internal HR documents (salary/bonus regulation, labor rules, collective labor agreement),
plus a benchmark of RAG techniques (chunking, embeddings, reranking, LLM-wiki, LangGraph/LangChain/LlamaIndex/Haystack/DSPy)
using a local Qwen3.5 VLM served by llama.cpp.

## Setup
- Python 3.12 via `uv sync`
- Put the original `.doc` files in `RAG/` (not tracked: internal documents)
- `soffice --headless --convert-to docx --outdir data/docx RAG/*.doc`
- `uv run python -m bibica_rag.ingest.parse_docx` → `data/parsed/` (Markdown + `nodes.jsonl`)

Models (`models/`) and llama.cpp binaries (`tools/`) are downloaded locally and not tracked.

## Web UI (PO extraction + RAG chat)

One Gradio app with two tabs sharing the same local Qwen VLM server:
- **📑 Trích xuất PO** — upload customer PO PDFs, get exactly the fields that customer needs (from the Excel spec),
  review them next to the rendered pages, download Excel/JSON.
- **💬 Hỏi đáp tài liệu nội bộ** — chat over the parsed HR documents (BM25 baseline retrieval + cited answers).

```bash
uv sync
MODEL=models/<qwen-vlm>.gguf MMPROJ=models/<mmproj>.gguf scripts/serve_qwen_vlm.sh   # llama-server on :8080
uv run python -m bibica_rag.app                                                         # http://127.0.0.1:7860
```
Env: `BIBICA_LLM_BASE_URL` (default `http://127.0.0.1:8080/v1`), `BIBICA_LLM_MODEL`, `BIBICA_PORT`.
The VLM needs its `mmproj` file to see the page images; any Qwen VL GGUF served by llama.cpp works (e.g. a 9B one).

## PO extraction (`bibica_rag/po/`)

Data (not tracked): `PO Extraction/` with the customer PDFs (sub-folders per customer are fine) and the Excel file
listing the fields to extract for each customer.

1. **Spec** (`spec.py`) — reads the Excel file; long (customer, field), matrix (fields × customers with `x`),
   wide (customers × fields) and sheet-per-customer layouts are detected automatically. Optional columns:
   description, type (text/number/date/table), group (fields of a line-item group become one table), example,
   alias (keywords to recognise the customer). Check what was understood:
   `uv run python -m bibica_rag.po.spec "PO Extraction/<file>.xlsx"`. If something is off, write the fixed
   version with `--yaml-out "PO Extraction/<file>.yaml"` (same stem as the Excel): it is merged on top.
2. **Customer** — from the file/folder name, else the PDF text layer, else the VLM picks from the customer list;
   can also be forced in the UI / with `--customer`.
3. **Extraction** (`extract.py`) — page images (+ text layer as a hint) → Qwen VLM, constrained by a JSON schema
   whose keys are exactly that customer's fields; output is filtered to the spec, nulls for absent values,
   numbers/dates normalized. `uv run python -m bibica_rag.po.extract "PO Extraction/" --out results/po`

### Testing
- Unit tests (no model or data needed: synthetic Excel/PDF + fake LLM): `uv run pytest`
- Accuracy on the real POs (`evaluate.py`):
  ```bash
  uv run python -m bibica_rag.po.extract "PO Extraction/" --out results/po
  uv run python -m bibica_rag.po.evaluate init-gt --pred results/po --gt data/po_gt   # then correct by hand, set "checked": true
  uv run python -m bibica_rag.po.evaluate score --pred results/po --gt data/po_gt     # results/po_eval/report.md
  BIBICA_LIVE=1 uv run pytest tests/test_po_live.py -s                                 # gate: customer 100%, exact field set, accuracy >= BIBICA_PO_MIN_ACC (0.9)
  ```
