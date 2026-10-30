"""Bibica web UI: PO extraction (VLM) + internal-document RAG chat, sharing one local Qwen server.

    uv run python -m bibica_rag.app            # http://127.0.0.1:7860
"""
import json
import os
import re
import tempfile
from pathlib import Path

import gradio as gr

from bibica_rag import llm
from bibica_rag.po.dms import load_dms_mapping
from bibica_rag.po.extract import extract_po
from bibica_rag.po.pdf import load_pdf
from bibica_rag.po.spec import CustomerSpec, default_spec_path, dump_specs, find_dms_file, load_specs
from bibica_rag.rag import chat

AUTO = "(Tự động nhận diện)"


# ---------------------------------------------------------------- PO extraction

def _load_specs(spec_file) -> tuple[dict[str, CustomerSpec], str]:
    path = Path(spec_file) if spec_file else default_spec_path()
    if not path:
        return {}, "⚠️ Chưa có file cấu hình trường: `config/po_customers.yaml`."
    try:
        specs = load_specs(path)
    except Exception as e:  # noqa: BLE001
        return {}, f"⚠️ Không đọc được `{path.name}`: {e}"
    n_fields = sum(len(s.fields) for s in specs.values())
    return specs, f"📄 `{path.name}` — {len(specs)} khách hàng, {n_fields} trường"


def _load_dms(dms_file) -> tuple[dict | None, str]:
    path = Path(dms_file) if dms_file else find_dms_file()
    if not path:
        return None, "⚠️ Chưa có file mã DMS (DS_SIEU_THI_VA_MA_DMS_*.xlsx) — sẽ không tra mã DMS."
    try:
        mapping = load_dms_mapping(path)
    except Exception as e:  # noqa: BLE001
        return None, f"⚠️ Không đọc được `{path.name}`: {e}"
    return mapping, f"🏪 `{path.name}` — {sum(len(s.rows) for s in mapping.values())} mã cửa hàng"


def on_dms_change(dms_file):
    return _load_dms(dms_file)


def on_spec_change(spec_file):
    specs, msg = _load_specs(spec_file)
    return specs, msg, gr.update(choices=[AUTO, *specs], value=AUTO), dump_specs(specs) if specs else ""


def run_extraction(pdf_files, specs, dms, customer, max_pages, use_text, progress=gr.Progress()):
    if not pdf_files:
        raise gr.Error("Chưa chọn file PDF")
    if not specs:
        raise gr.Error("Chưa có cấu hình khách hàng (file Excel)")
    results = []
    for path in progress.tqdm(pdf_files, desc="Đang trích xuất"):
        r = extract_po(path, specs, customer=None if customer in (None, AUTO) else customer,
                       max_pages=int(max_pages), use_text_layer=use_text, dms_mapping=dms)
        r.file = Path(path).name
        results.append(r.to_dict())
    names = [r["file"] for r in results]
    return (results, _summary(results), gr.update(choices=names, value=names[0]),
            *show_result(names[0], results, pdf_files), export(results))


def _summary(results):
    rows = []
    for r in results:
        n = len(r["fields"])
        dms = r.get("dms") or {}
        rows.append([r["file"], r["customer"] or "", r["customer_reason"],
                     f"{n - len(r['missing'])}/{n}" if n else "", dms.get("Mã DMS") or dms.get("ghi_chu", ""),
                     r["elapsed_s"], r["error"]])
    return rows


def show_result(name, results, pdf_files):
    r = next((x for x in results or [] if x["file"] == name), None)
    if r is None:
        return [], {}, []
    table = []
    for k, v in r["fields"].items():
        table.append([k, json.dumps(v, ensure_ascii=False, indent=1) if isinstance(v, (list, dict)) else v])
    for k, v in (r.get("dms") or {}).items():
        if v:
            table.append([f"[DMS] {k}", v])
    images = []
    for p in pdf_files or []:
        if Path(p).name == name:
            pages = load_pdf(p, dpi=100, max_pages=30).images
            images = [(img, f"Trang {i + 1}" + (" ✓" if i + 1 in r.get("pages_used", []) else " (bỏ qua)"))
                      for i, img in enumerate(pages)]
    out = {"customer": r["customer"], "fields": r["fields"], "dms": r.get("dms", {})}
    if r["error"]:
        out["error"] = r["error"]
    return table, out, images


def _sheet_name(s: str, used: set) -> str:
    base = re.sub(r"[\[\]:*?/\\]", "_", s)[:31] or "Sheet"
    name, i = base, 1
    while name in used:
        i += 1
        name = f"{base[:28]}_{i}"
    used.add(name)
    return name


def export(results) -> list[str]:
    """JSON (everything) + Excel (one sheet per customer, one per line-item table)."""
    import pandas as pd

    out_dir = Path(tempfile.mkdtemp(prefix="po_"))
    js = out_dir / "po_extraction.json"
    js.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    sheets = {}
    for r in results:
        if not r["customer"]:
            continue
        head = {"File": r["file"]}
        for k, v in r["fields"].items():
            if isinstance(v, list):
                sheets.setdefault(f"{r['customer']} - {k}", []).extend({"File": r["file"], **row} for row in v)
            else:
                head[k] = v
        for k in ("Mã DMS", "Tên siêu thị DMS", "Đơn vị tính DMS"):
            if (r.get("dms") or {}).get(k):
                head[k] = r["dms"][k]
        sheets.setdefault(r["customer"], []).append(head)
    xl = out_dir / "po_extraction.xlsx"
    with pd.ExcelWriter(xl) as w:
        used = set()
        for name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(w, sheet_name=_sheet_name(name, used), index=False)
        if not sheets:
            pd.DataFrame(_summary(results)).to_excel(w, sheet_name="Summary", index=False)
    return [str(xl), str(js)]


# ---------------------------------------------------------------- RAG chat

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        if not chat.NODES.exists():
            raise gr.Error("Chưa có dữ liệu RAG: chạy `python -m bibica_rag.ingest.parse_docx` trước")
        _retriever = chat.Retriever.from_jsonl(chat.NODES)
    return _retriever


def sources_md(hits) -> str:
    if not hits:
        return "_Không tìm thấy trích đoạn liên quan._"
    out = []
    for h in hits:
        snippet = h.node["text"][:400].replace("\n", " ")
        out.append(f"**{h.node['node_id']}** — {h.node.get('title', '')} _(BM25 {h.score:.1f})_\n\n> {snippet}…")
    return "\n\n".join(out)


def respond(message, history, k):
    if not message.strip():
        yield history, "", gr.update()
        return
    retriever = get_retriever()
    prior = list(history)
    history = prior + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    yield history, "", gr.update()
    try:
        for text, hits in chat.answer_stream(message, retriever, prior, k=int(k)):
            history[-1]["content"] = text
            yield history, "", sources_md(hits)
    except Exception as e:  # noqa: BLE001
        history[-1]["content"] = f"⚠️ Lỗi gọi LLM: {e}"
        yield history, "", gr.update()


# ---------------------------------------------------------------- layout

def build_app() -> gr.Blocks:
    specs0, msg0 = _load_specs(None)
    dms0, dms_msg0 = _load_dms(None)
    with gr.Blocks(title="Bibica AI") as app:
        gr.Markdown("# Bibica AI — Trích xuất PO & Hỏi đáp tài liệu nội bộ")
        with gr.Row():
            status = gr.Markdown(llm.server_status())
            gr.Button("Kiểm tra LLM", size="sm", scale=0).click(llm.server_status, outputs=status)

        with gr.Tab("📑 Trích xuất PO"):
            specs = gr.State(specs0)
            dms = gr.State(dms0)
            results = gr.State([])
            with gr.Row():
                with gr.Column(scale=1):
                    pdfs = gr.File(label="File PO (PDF)", file_count="multiple", file_types=[".pdf"],
                                   type="filepath")
                    with gr.Accordion("File cấu hình", open=False):
                        spec_file = gr.File(label="Trường cần trích xuất (YAML) — trống: config/po_customers.yaml",
                                            file_types=[".yaml", ".yml", ".json", ".xlsx"], type="filepath")
                        dms_file = gr.File(label="Mã siêu thị → mã DMS (Excel) — trống: file trong PO Extraction/",
                                           file_types=[".xlsx", ".xlsm"], type="filepath")
                    spec_msg = gr.Markdown(msg0)
                    dms_msg = gr.Markdown(dms_msg0)
                    customer = gr.Dropdown([AUTO, *specs0], value=AUTO, label="Khách hàng")
                    with gr.Accordion("Tùy chọn", open=False):
                        max_pages = gr.Slider(1, 30, value=20, step=1, label="Số trang tối đa đọc từ PDF")
                        use_text = gr.Checkbox(True, label="Gửi kèm lớp text của PDF (nếu có)")
                    btn = gr.Button("Trích xuất", variant="primary")
                with gr.Column(scale=2):
                    summary = gr.Dataframe(headers=["File", "Khách hàng", "Nhận diện", "Đã điền", "Mã DMS", "Giây",
                                                    "Lỗi"],
                                           label="Kết quả", interactive=False, wrap=True)
                    pick = gr.Dropdown([], label="Xem chi tiết file")
                    with gr.Row():
                        fields = gr.Dataframe(headers=["Trường", "Giá trị"], label="Thông tin trích xuất",
                                              interactive=False, wrap=True)
                        gallery = gr.Gallery(label="PDF", columns=1, height=620, object_fit="contain")
                    with gr.Accordion("JSON", open=False):
                        js = gr.JSON()
                    files_out = gr.File(label="Tải kết quả", file_count="multiple")
            with gr.Accordion("Các trường cần trích xuất theo khách hàng (đọc từ file cấu hình)", open=False):
                spec_view = gr.Code(dump_specs(specs0) if specs0 else "", language="yaml")

            spec_file.change(on_spec_change, spec_file, [specs, spec_msg, customer, spec_view])
            dms_file.change(on_dms_change, dms_file, [dms, dms_msg])
            btn.click(run_extraction, [pdfs, specs, dms, customer, max_pages, use_text],
                      [results, summary, pick, fields, js, gallery, files_out])
            pick.change(show_result, [pick, results, pdfs], [fields, js, gallery])

        with gr.Tab("💬 Hỏi đáp tài liệu nội bộ"):
            with gr.Row():
                with gr.Column(scale=3):
                    bot = gr.Chatbot(label="Trợ lý nhân sự Bibica", height=560)
                    q = gr.Textbox(placeholder="Ví dụ: Nghỉ phép năm được bao nhiêu ngày?", show_label=False)
                    with gr.Row():
                        k = gr.Slider(1, 10, value=5, step=1, label="Số điều khoản truy xuất (top-k)")
                        gr.Button("Xóa hội thoại", scale=0).click(lambda: ([], ""), outputs=[bot, q])
                with gr.Column(scale=2):
                    gr.Markdown("### Nguồn trích dẫn")
                    src = gr.Markdown()
            q.submit(respond, [q, bot, k], [bot, q, src])
    return app


def main():
    build_app().queue().launch(server_name=os.environ.get("BIBICA_HOST", "127.0.0.1"),
                               server_port=int(os.environ.get("BIBICA_PORT", "7860")))


if __name__ == "__main__":
    main()
