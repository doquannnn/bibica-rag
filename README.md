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

Data (not tracked): `PO Extraction/` with the retailer PO PDFs and `DS_SIEU_THI_VA_MA_DMS_*.xlsx`.

1. **Fields per retailer** — `config/po_customers.yaml`, transcribed from the annotated PO screenshots in the
   Excel (one sheet per retailer): Số PO, Ngày đề nghị giao hàng, Mã KH and the line items (Mã SP, SL đặt; BHX also
   Đơn vị), with where each value is printed on that retailer's PO. `offset_days: 5` implements "lấy ngày này + 5
   ngày" (BigC, Lan Chi, Satra); `skip_pages_containing` drops BigC's "DETAIL DELIVERY BY STORE" pages.
   Print it with `uv run python -m bibica_rag.po.spec`.
2. **Customer** — from the file/folder name, else the PDF text layer, else the VLM picks from the list;
   can also be forced in the UI / with `--customer`.
3. **Extraction** (`extract.py`) — one VLM call per page (page 1: all fields; later pages: line items only,
   concatenated), constrained by a JSON schema whose keys are exactly that retailer's fields; the PDF text layer
   is added as a hint unless it is garbled (e.g. AEON's fonts). Output is filtered to the spec, nulls for absent
   values, numbers/dates normalized.
4. **DMS code** (`dms.py`) — Mã KH is looked up in the Excel (MÃ TRÊN PO PDF → MÃ DMS, TÊN SIÊU THỊ DMS,
   "Đơn vị tính ..."); unknown codes are reported so the Excel can be completed.

`uv run python -m bibica_rag.po.extract "PO Extraction/" --out results/po`

### Testing
- Unit tests (no model or data needed: synthetic Excel/PDF + fake LLM): `uv run pytest`
- Accuracy on the real POs (`evaluate.py`):
  ```bash
  uv run python -m bibica_rag.po.extract "PO Extraction/" --out results/po
  uv run python -m bibica_rag.po.evaluate init-gt --pred results/po --gt data/po_gt   # then correct by hand, set "checked": true
  uv run python -m bibica_rag.po.evaluate score --pred results/po --gt data/po_gt     # results/po_eval/report.md
  BIBICA_LIVE=1 uv run pytest tests/test_po_live.py -s                                 # gate: customer 100%, exact field set, accuracy >= BIBICA_PO_MIN_ACC (0.9)
  ```
  `tests/test_po_live.py::test_real_po_preprocessing` runs without the model whenever the PDFs and GT are present.
