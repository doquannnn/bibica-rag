# PO Extraction — brief for the cloud session

This file is the starting context for a session running in the cloud, which has **only what is on GitHub** (no local models,
no `RAG/`, no `models/`). Everything it needs about the task is here or in `PO Extraction/`.

## Goal
Supermarket purchase orders (PO) arrive as PDF. Today GSBH (sales supervisors) type each order into the DMS by hand.
Build a feature where they **upload the PO PDF and the system extracts the order and records it in the DMS automatically**
(wording from the `HUONG DAN` sheet of the workbook). Extraction must use a cloud model (Claude vision), not the local Qwen.

## Data in this repo (`PO Extraction/`)
| File | What it is |
|---|---|
| `Aeon ok.pdf`, `BHX ok.pdf`, `BigC ok.pdf`, `Coop ok.pdf`, `EMART ok.PDF`, `Lotte ok.pdf`, `Mega.pdf`, `VIn.pdf`, `satra 3.PDF` | One real sample PO per chain (Oct–Nov 2023). Layouts differ per chain; BigC has 10 pages. |
| `DS SIEU THI VA MA DMS 230530.xlsx` | Mapping workbook (see below) |

Observed (via `pdftotext` / `pdfimages`, not verified further):
- Text-layer PDFs: BHX, BigC, Coop, Lotte, VIn, EMART, Satra.
- **Aeon** and **Mega** appear to be image-based (Aeon text layer is garbled; Mega has one large JPEG, no text) → need vision/OCR.
- The buyer/supplier on samples is "CÔNG TY CỔ PHẦN PHÂN PHỐI HÀNG TIÊU DÙNG PAN" (Pan), which is the distributor.
- No ground-truth extraction exists yet. Building one (hand-checked JSON per sample) is the first step of any evaluation.

### Workbook structure
- `HUONG DAN`: purpose + a table of 10 chains with an example store code each (Aeon, BHX, BigC, Coop, Emart, Lotte,
  Mega/"MM", Satra, Vin, Lan Chi). Column "HÌNH ĐƠN HÀNG" is meant to hold the PO image.
- `DANH SACH`: store name → store code on the PO (`MÃ ST`) → DMS customer code (`MÃ DMS`). Only partly filled.
- One sheet per chain (`Lotte`, `BHX`, `E-Mart`, `MM`, `LANCHI`, `SATRA`, `VIN`, `BigC`, `AEON`, `COOP`) with columns
  `TÊN SIÊU THỊ | MÃ TRÊN PO PDF | MÃ DMS | TÊN SIÊU THỊ DMS` (store-code mapping) and a note on the **unit of measure**
  used by that chain's PO ("Thùng" for Lotte/BigC/Aeon/Coop, "túi hộp" for BHX/MM). Some sheets also carry product mapping
  columns further right. LANCHI notes "Không có số PO" (no PO number). Read the sheets directly; do not trust this summary for details.

## What to build
1. **Extractor**: PDF → structured JSON. Suggested schema (confirm with the owner):
   `chain, po_number, order_date, delivery_date, store_code_on_po, store_name, supplier, lines[{po_item_code, product_name, unit, quantity, unit_price, amount}], total`.
   Send PDF pages as images (and the text layer when present) to a Claude vision-capable model; require JSON-only output and validate with a schema.
2. **Mapping layer**: store code on PO → `MÃ DMS` using the workbook; product code → DMS product code; normalise units
   (thùng vs túi/hộp) per chain rules from the workbook. Unmapped codes must be reported, never guessed.
3. **Evaluation**: hand-verify the 9 samples as gold, report field-level accuracy (header fields, per-line fields, totals).
   Cross-check `sum(lines.amount)` against the PO total as an automatic sanity check.
4. **Output**: DMS API details are unknown. Until provided, emit a DMS-ready JSON/CSV and stop there.

## Reusable code already in the repo
- `bibica_rag/llm.py`: OpenAI-compatible client with disk cache; `image_part(path)` builds an image message part.
  Point `LLM_BASE_URL` / `LLM_MODEL` at any OpenAI-compatible endpoint. For Claude, use the Anthropic SDK instead.
- `app.py`: Gradio UI with an image tab (`ask_image`) and a default JSON-extraction prompt — a starting point for the upload UI.
- `bibica_rag/eval/keyfacts.py`: number/date normalisation helpers usable for field matching.
- The RAG benchmark itself (HR documents) is unrelated; see `README.md` and `results/report.html`.

## Constraints and cautions
- The repo is **private** and the PO samples are real commercial documents (prices, store codes). Do not copy them into public places or logs.
- Local-only items not in the repo: `models/`, `tools/`, `RAG/`, `data/parsed/`, `data/cache/`.
- Open questions for the owner (do not assume): the DMS integration interface; product code mapping source; exact
  rules for unit conversion; whether orders can span several delivery stores in one PDF; required fields in DMS.
