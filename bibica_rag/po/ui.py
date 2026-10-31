"""Gradio tab "Trích xuất PO": upload retailer PO PDFs, get exactly that retailer's fields, review, download.

Used by the root `app.py` (`build_po_tab()` inside a `gr.Tab`); can also run alone:
    uv run python -m bibica_rag.po.ui
"""
import json
import re
import tempfile
from pathlib import Path

import gradio as gr

from bibica_rag.po.dms import load_dms_mapping
from bibica_rag.po.extract import extract_po
from bibica_rag.po.pdf import load_pdf
from bibica_rag.po.spec import CustomerSpec, default_spec_path, dump_specs, find_dms_file, load_specs
from bibica_rag.po.vlm import backend_from_env

AUTO = "(Tự động nhận diện)"
PROVIDERS = ["local", "gemini"]


def _load_specs(spec_file) -> tuple[dict[str, CustomerSpec], str]:
    path = Path(spec_file) if spec_file else default_spec_path()
    if not path:
        return {}, "⚠️ Chưa có file cấu hình trường: `config/po_customers.yaml`."
    try:
        specs = load_specs(path)
    except Exception as e:  # noqa: BLE001
        return {}, f"⚠️ Không đọc được `{path.name}`: {e}"
    return specs, f"📄 `{path.name}` — {len(specs)} khách hàng"


def _load_dms(dms_file) -> tuple[dict | None, str]:
    path = Path(dms_file) if dms_file else find_dms_file()
    if not path:
        return None, "⚠️ Chưa có file mã DMS — sẽ không tra mã DMS."
    try:
        mapping = load_dms_mapping(path)
    except Exception as e:  # noqa: BLE001
        return None, f"⚠️ Không đọc được `{path.name}`: {e}"
    return mapping, f"🏪 `{path.name}` — {sum(len(s.rows) for s in mapping.values())} mã cửa hàng"


def output_view(r: dict) -> dict:
    """What the user needs: the customer's fields (+ DMS code); diagnostics stay in the full JSON export."""
    out = {"Khách hàng": r["customer"], **r["fields"]}
    dms = r.get("dms") or {}
    if dms.get("Mã DMS"):
        out["Mã DMS"] = dms["Mã DMS"]
    return out


def run_extraction(pdf_files, specs, dms, customer, provider, use_text, local_model=None, progress=gr.Progress()):
    if not pdf_files:
        raise gr.Error("Chưa chọn file PDF")
    if not specs:
        raise gr.Error("Chưa có cấu hình khách hàng")
    backend = backend_from_env(provider, local_model=local_model)
    results = []
    for path in progress.tqdm(pdf_files, desc="Đang trích xuất"):
        r = extract_po(path, specs, customer=None if customer in (None, AUTO) else customer, max_pages=30,
                       use_text_layer=use_text, dms_mapping=dms, backend=backend)
        r.file = Path(path).name
        results.append(r.to_dict())
    names = [r["file"] for r in results]
    return (results, summary_rows(results), gr.update(choices=names, value=names[0]),
            *show_result(names[0], results, pdf_files), export(results))


def summary_rows(results):
    rows = []
    for r in results:
        items = next((v for v in r["fields"].values() if isinstance(v, list)), [])
        dms = r.get("dms") or {}
        rows.append([r["file"], r["customer"] or "", len(items), dms.get("Mã DMS") or "",
                     len(r.get("warnings", [])), r["elapsed_s"], r["error"]])
    return rows


def show_result(name, results, pdf_files):
    r = next((x for x in results or [] if x["file"] == name), None)
    if r is None:
        return [], [], {}, "", []
    head = [[k, v] for k, v in output_view(r).items() if not isinstance(v, list)]
    items = next((v for v in r["fields"].values() if isinstance(v, list)), [])
    lines = [list(row.values()) for row in items]
    warn = "\n".join(f"- ⚠️ {w}" for w in r.get("warnings", [])) or "✅ Không có cảnh báo tự kiểm tra."
    if r["error"]:
        warn = f"❌ {r['error']}\n\n" + warn
    images = []
    for p in pdf_files or []:
        if Path(p).name == name:
            pages = load_pdf(p, dpi=90, max_pages=30).images
            images = [(img, f"Trang {i + 1}" + (" ✓" if i + 1 in r.get("pages_used", []) else " (bỏ qua)"))
                      for i, img in enumerate(pages)]
    return head, gr.update(value=lines, headers=list(items[0]) if items else None), output_view(r), warn, images


def _sheet_name(s: str, used: set) -> str:
    base = re.sub(r"[\[\]:*?/\\]", "_", s)[:31] or "Sheet"
    name, i = base, 1
    while name in used:
        i += 1
        name = f"{base[:28]}_{i}"
    used.add(name)
    return name


def export(results) -> list[str]:
    """Excel (one sheet of headers + one of lines per customer) and JSON (only the extracted information)."""
    import pandas as pd

    out_dir = Path(tempfile.mkdtemp(prefix="po_"))
    js = out_dir / "po_extraction.json"
    js.write_text(json.dumps([{"file": r["file"], **output_view(r)} for r in results if r["customer"]],
                             ensure_ascii=False, indent=2), encoding="utf-8")
    sheets = {}
    for r in results:
        if not r["customer"]:
            continue
        head = {"File": r["file"]}
        for k, v in output_view(r).items():
            if isinstance(v, list):
                po = r["fields"].get("Số PO")
                sheets.setdefault(f"{r['customer']} - {k}", []).extend(
                    {"File": r["file"], **({"Số PO": po} if po else {}), **row} for row in v)
            elif k != "Khách hàng":
                head[k] = v
        sheets.setdefault(r["customer"], []).append(head)
    xl = out_dir / "po_extraction.xlsx"
    with pd.ExcelWriter(xl) as w:
        used = set()
        for name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(w, sheet_name=_sheet_name(name, used), index=False)
        if not sheets:
            pd.DataFrame(summary_rows(results)).to_excel(w, sheet_name="Summary", index=False)
    return [str(xl), str(js)]


def build_po_tab(local_model=None):
    """Components of the PO tab; call inside `with gr.Tab(...)` of a gr.Blocks.

    `local_model`: optional gr.Dropdown of bibica_rag.llm.ENDPOINTS (app.py's model switch) used by provider "local"."""
    specs0, msg0 = _load_specs(None)
    dms0, dms_msg0 = _load_dms(None)
    default_provider = backend_from_env().provider
    specs, dms, results = gr.State(specs0), gr.State(dms0), gr.State([])
    with gr.Row():
        with gr.Column(scale=1):
            pdfs = gr.File(label="File PO (PDF)", file_count="multiple", file_types=[".pdf"], type="filepath")
            customer = gr.Dropdown([AUTO, *specs0], value=AUTO, label="Khách hàng")
            provider = gr.Dropdown(PROVIDERS, value=default_provider if default_provider in PROVIDERS else "local",
                                   label="VLM (local: model chọn ở đầu trang, mặc định Qwen3.5-4B — scripts/serve.sh 4B · "
                                         "gemini: cần GEMINI_API_KEY)")
            btn = gr.Button("Trích xuất PO", variant="primary")
            with gr.Accordion("Cấu hình", open=False):
                use_text = gr.Checkbox(True, label="Gửi kèm lớp chữ của PDF (nếu đọc được)")
                spec_file = gr.File(label="Trường cần trích xuất (YAML) — trống: config/po_customers.yaml",
                                    file_types=[".yaml", ".yml", ".json"], type="filepath")
                dms_file = gr.File(label="Mã siêu thị → mã DMS (Excel) — trống: file trong 'PO Extraction/'",
                                   file_types=[".xlsx", ".xlsm"], type="filepath")
                spec_msg, dms_msg = gr.Markdown(msg0), gr.Markdown(dms_msg0)
                spec_view = gr.Code(dump_specs(specs0) if specs0 else "", language="yaml",
                                    label="Các trường theo khách hàng")
        with gr.Column(scale=2):
            summary = gr.Dataframe(headers=["File", "Khách hàng", "Số dòng", "Mã DMS", "Cảnh báo", "Giây", "Lỗi"],
                                   label="Kết quả", interactive=False, wrap=True)
            pick = gr.Dropdown([], label="Xem chi tiết file")
            warn = gr.Markdown()
            with gr.Row():
                with gr.Column():
                    head = gr.Dataframe(headers=["Trường", "Giá trị"], label="Thông tin đơn hàng",
                                        interactive=False, wrap=True)
                    lines = gr.Dataframe(label="Chi tiết hàng", interactive=False, wrap=True)
                gallery = gr.Gallery(label="PDF", columns=1, height=600, object_fit="contain")
            with gr.Accordion("JSON", open=False):
                js = gr.JSON()
            files_out = gr.File(label="Tải kết quả (Excel / JSON)", file_count="multiple")

    def on_spec(f):
        s, m = _load_specs(f)
        return s, m, gr.update(choices=[AUTO, *s], value=AUTO), dump_specs(s) if s else ""

    spec_file.change(on_spec, spec_file, [specs, spec_msg, customer, spec_view])
    dms_file.change(_load_dms, dms_file, [dms, dms_msg])
    btn.click(run_extraction, [pdfs, specs, dms, customer, provider, use_text] + ([local_model] if local_model else []),
              [results, summary, pick, head, lines, js, warn, gallery, files_out])
    pick.change(show_result, [pick, results, pdfs], [head, lines, js, warn, gallery])


if __name__ == "__main__":
    with gr.Blocks(title="Bibica PO") as demo:
        gr.Markdown("## Bibica — Trích xuất đơn đặt hàng (PO) siêu thị")
        build_po_tab()
    demo.queue().launch(server_name="127.0.0.1", server_port=7861)
