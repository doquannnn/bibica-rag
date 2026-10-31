# PO extraction

Supermarket purchase orders (PO) arrive as PDF; GSBH type each one into the DMS by hand. This feature reads the PDF
and returns **only the information Bibica needs for that retailer** (from `PO Extraction/DS SIEU THI VA MA DMS 230530.xlsx`),
plus the DMS customer code. DMS API details are unknown, so the output stops at Excel/JSON.

## What is extracted per retailer
Source: the annotated screenshots and text boxes of each retailer sheet in the Excel (the field list is in
pictures, not cells, so it is transcribed in `config/po_customers.yaml`; print it with `python -m bibica_rag.po.spec`).

| Retailer | Số PO | Ngày đề nghị giao hàng | Mã KH (store code) | Line items |
|---|---|---|---|---|
| AEON | Số đơn hàng | Ngày Giao Hàng | Mã Cửa Hàng | Mã Vạch, SL Đặt |
| BHX | Mã đơn hàng | Ngày NCC giao hàng (first day of the range) | number of "Kho NCC giao hàng" | Mã sản phẩm, SL, ĐVT |
| BigC | Order No | Order Date **+ 5 days** | Com. Contract | Article, OU Qty — only "PURCHASE NOTE" pages, not "DETAIL DELIVERY BY STORE" |
| COOP | P/O Number | Expected Receipt Date | P/O Location | SKU Number, Qty Ord/CS |
| E-Mart | PO No. | Delivery Date | Delivery to (DMS code matched on "366 PHAN") | Unit Barcode, PO Qty. |
| Lotte | Ord slip no | Sply dt | Str/ nm | Sale cd, Ord qty |
| MM (Mega) | Số thứ tự đơn đặt hàng | Giao hàng ngày | ILN of "Địa điểm giao hàng" | EAN sản phẩm, Đơn đặt hàng số lượng |
| SATRA | Đơn đặt hàng số | "Ngày" (month/day/year) **+ 5 days** | Mã số NCC | Barcode, Số lượng (decimal comma: 1,000 = 1) |
| VIN | Số đơn hàng (PO No.) | Ngày giao (Delivery Date) | number of the delivery address (1291-WMP_...) | Mã vạch, Số lượng |
| LANCHI | — (no PO number) | Ngày đặt **+ 5 days** | Tên siêu thị | Mã vạch, Tổng đặt — *no sample PDF yet* |

## Pipeline (`bibica_rag/po/`)
1. `pdf.py` — render pages (150 dpi) + text layer; `extract.text_is_usable` drops garbled layers (AEON fonts) and
   scans (Mega has none).
2. `spec.detect_customer` — retailer from the file name, else the text layer, else the VLM chooses from the list.
3. `extract.py` — one VLM call per page with that retailer's field list and descriptions (where each value is printed,
   which look-alike values not to take), JSON-schema constrained when the server supports it. Page 1 gives every
   field, later pages only line items. The text layer is sent as a hint for digital PDFs.
4. Post-processing: keep exactly the spec's keys, normalize numbers/dates (`date_order: mdy`, `decimal_comma`),
   add `offset_days`.
5. `dms.py` — store code → MÃ DMS from the retailer's sheet; unknown codes are reported, never guessed.
6. `validate.py` — self-checks without ground truth, shown as warnings to the reviewer (see below).

VLM backend (`vlm.py`): by default the **local Qwen3.5-4B VLM** (`scripts/serve.sh 4B`, model `qwen3.5-4B` on
:8080); optionally the **Gemini API** (`BIBICA_PO_PROVIDER=gemini GEMINI_API_KEY=...`, default `gemini-3.5-flash`); any other OpenAI-compatible
endpoint with `BIBICA_PO_PROVIDER=openai BIBICA_PO_BASE_URL=... BIBICA_PO_MODEL=... BIBICA_PO_API_KEY=...`.
The cloud session could not download a local model (Hugging Face is blocked by its network policy), so the cloud
path is Gemini. Note: on Google's free tier, prompts may be used to improve Google products — the POs contain
commercial data; use a paid key or the local model if that is not acceptable.

## How it is evaluated
Extraction is a structured-output task: the question is not "is the text similar" but "would this order go into the
DMS correctly". So:

- **Ground truth** `data/po_gt/*.json`: the 9 samples, checked by hand against the rendered pages. Built from the text
  layer for digital PDFs and read from the image for AEON/Mega; every GT passes the self-checks below, and line
  quantities add up to the total printed on the PO (BHX 420, BigC 197, COOP 639, Lotte 16, Mega 484, Satra 26,
  VIN 1635) — `tests/test_po_live.py::test_gt_is_grounded`.
- **Metrics** (`evaluate.py`):
  - *order exact match* — every header field and every line exactly right: the share of POs needing no correction
    (the business metric);
  - header field accuracy (strict after type normalization; loose ignoring accents/punctuation);
  - line items **matched by product code**, not by position (one skipped line must not make every following line
    wrong): line precision, recall, F1; per-column accuracy on matched lines (quantity); lines fully correct;
  - customer detection accuracy; spec conformity (exactly the retailer's fields, nothing else).
  The report lists each error (missed / extra / wrong line, wrong header value) per file.
- **Self-checks at run time** (`validate.py`, no GT needed — this is what protects production orders):
  values grounded in the PDF text layer (code present, quantity printed near its code, PO number / store code
  present), EAN-13 check digit (a misread digit breaks it), duplicate lines, missing fields, store code not in the DMS
  Excel. Scans (AEON, Mega) get only the mechanical checks, so they deserve a closer human look — or a second model
  run and comparison.
- **Tests**: unit tests with fake clients and synthetic PDFs/Excel; model-free tests on the real samples (customer
  detection, BigC page selection, GT grounding, DMS codes); an end-to-end run on the 9 real PDFs with a scripted VLM
  that copies values as printed (proves page selection, +5 days, month-first dates, decimal commas and the evaluator
  give 100% when the model reads correctly); and the live gate `BIBICA_LIVE=1 pytest tests/test_po_live.py -s`
  (customer 100%, exact field set, header accuracy and line F1 ≥ `BIBICA_PO_MIN_ACC`, default 0.9).

## Results (2026-10-31, the 9 samples: 87 line items)
| Run | Orders fully correct | Header fields | Line F1 | Qty on matched lines |
|---|---|---|---|---|
| `gemini-3.5-flash-lite`, image + text layer | 9/9 | 27/27 | 100% | 100% |
| `gemini-3.5-flash-lite`, image only (`--no-text-layer`, simulates scans) | 9/9 | 27/27 | 100% | 100% |
| `gemini-3.5-flash`, image + text layer | 3/3 completed (Aeon, BHX, COOP); the other 6 hit the free-tier daily quota | | | |

Read with care: 9 documents, one per retailer, so this shows the field rules and the pipeline work on each layout;
it is not a statistical accuracy estimate. Every new PO should be added to `data/po_gt/` once checked, and the
self-check warnings stay the safety net in production. A first `gemini-3.5-flash` run also scored 9/9 but is
discarded: the prompt examples then contained values copied from the samples
(`test_prompt_examples_do_not_leak_the_ground_truth` now prevents this).

Gemini free tier: **20 requests per day per model** (one request per page: 10 for the 9 samples). `gemini-2.5-flash`
is no longer offered to new keys; `gemini-3.8-flash` returned "high demand" (503) during the test.
Reproduce: `export GEMINI_API_KEY=...; python -m bibica_rag.po.extract "PO Extraction/" --model gemini-3.5-flash-lite
--out results/po/x; python -m bibica_rag.po.evaluate score --pred results/po/x`.

## Open points for the owner
- Rules applied as written in the Excel, though the PO prints another date: BigC "Order Date + 5" (09/11/23 → 14/11)
  while the PO says "Delivery Date To Store 15/11/23"; Satra "Ngày + 5" (11/28 → 03/12) while the PO says
  "Ngày giao hàng 11/29/2023". Confirm which one the DMS needs.
- Store codes of the samples missing from the Excel: AEON 8003, BHX 7211, COOP 801 (COOP sheet empty),
  Lotte 01011, VIN 1291 → the DMS code is reported as missing.
- Product code → DMS product code and unit conversion (thùng vs túi/hộp) are not mapped: the Excel has no product table.
- No Lan Chi sample PDF.
