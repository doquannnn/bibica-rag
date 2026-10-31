"""Simple UI: ask questions over the Bibica HR documents, or extract retailer purchase orders (PO PDF -> the
fields that retailer needs, see bibica_rag/po).

    scripts/serve.sh 4B            # llama-server on :8080 (or 9B)
    # PO extraction uses the same local Qwen3.5-4B VLM; optional: BIBICA_PO_PROVIDER=gemini GEMINI_API_KEY=...
    uv run python app.py           # http://127.0.0.1:7860 (other port: BIBICA_PORT=7861)
"""
import json
import os
import time
from functools import lru_cache

import gradio as gr
import yaml

from bibica_rag.corpus import ROOT
from bibica_rag.eval.run_pipelines import build_pipeline
from bibica_rag.llm import MODEL
from bibica_rag.po.ui import build_po_tab

CONFIGS = ROOT / "configs"


def available_pipelines():
    """Pipelines from the benchmark configs, best (per leaderboard) first."""
    specs = {}
    for p in sorted(CONFIGS.glob("*.yaml")):
        for s in yaml.safe_load(p.read_text(encoding="utf-8")) or []:
            specs.setdefault(s["run"], s)
    if not specs:
        specs = {"naive-clause-bm25": {"run": "naive-clause-bm25", "kind": "naive", "chunking": "clause_header",
                                       "retriever": "bm25:foldplus"}}
    order = []
    lb = ROOT / "results" / "leaderboard.json"
    if lb.exists():
        for r in json.loads(lb.read_text(encoding="utf-8")):
            run = r["run"].split("@")[0]
            if run in specs and run not in order:
                order.append(run)
    order += [k for k in specs if k not in order]
    # CAG needs the 100k-context server on :8081 (LLM_BASE_URL), so the default is the best pipeline for :8080
    order.sort(key=lambda k: specs[k]["kind"] == "cag")
    return order, specs


ORDER, SPECS = available_pipelines()


@lru_cache(maxsize=None)
def get_pipe(run):
    return build_pipeline(SPECS[run])


def ask_rag(question, run, history):
    history = history or []
    if not question.strip():
        return history, "", ""
    t0 = time.time()
    r = get_pipe(run).answer(question)
    history = history + [{"role": "user", "content": question}, {"role": "assistant", "content": r["answer"]}]
    t = r["timings"]
    info = (f"📚 **Nguồn tham khảo:** {', '.join(r.get('retrieved_nodes', [])[:8]) or 'toàn văn tài liệu'}  \n"
            f"⏱️ Trả lời trong {time.time() - t0:.1f} giây (tìm tài liệu {t.get('retrieval_s') or 0:.1f}s · "
            f"chữ đầu tiên {t.get('ttft_s') or 0:.1f}s) · {r.get('llm_calls', 1)} lần gọi mô hình · "
            f"`{run}` · `{MODEL}`")
    ctx = "\n\n---\n\n".join(f"**{', '.join(c['node_ids'])}**\n\n{c['context']}" for c in r.get("contexts", [])[:8])
    return history, info, ctx or "_Phương pháp này không tách đoạn — mô hình đọc toàn văn tài liệu._"


THEME = gr.themes.Soft(primary_hue="orange", neutral_hue="slate", font=[gr.themes.GoogleFont("Be Vietnam Pro"),
                                                                       "ui-sans-serif", "sans-serif"])
CSS = """
.gradio-container {max-width: 1200px !important; width: 100% !important; margin: auto;}
#header h1 {margin-bottom: 0.2em;}
#header p {color: var(--body-text-color-subdued); margin-top: 0;}
#meta {font-size: 0.88em; color: var(--body-text-color-subdued);}
table th, table td, table th *, table td * {font-family: var(--font) !important;}
"""
EXAMPLES = [
    "Làm thêm giờ vào ngày lễ được trả bao nhiêu phần trăm lương?",
    "Mức hỗ trợ hằng tháng cho công nhân trực tiếp sản xuất là bao nhiêu?",
    "Người lao động được nghỉ phép năm bao nhiêu ngày?",
]

with gr.Blocks(title="Bibica — Trợ lý tài liệu") as demo:
    gr.Markdown("# Bibica — Trợ lý tài liệu nội bộ\n"
                "Hỏi đáp về Quy chế lương thưởng, Nội quy lao động, Thỏa ước lao động tập thể "
                "và trích xuất đơn đặt hàng (PO) của siêu thị.", elem_id="header")
    with gr.Tab("💬 Hỏi đáp"):
        chatbot = gr.Chatbot(height=460, label="Hội thoại", show_label=False,
                             placeholder="Đặt câu hỏi về quy chế, nội quy hoặc thỏa ước. "
                                         "Câu trả lời luôn kèm trích dẫn điều khoản.")
        with gr.Row(equal_height=True):
            q = gr.Textbox(placeholder="Nhập câu hỏi rồi nhấn Enter…", show_label=False, lines=1, scale=8,
                           autofocus=True)
            send = gr.Button("Gửi", variant="primary", scale=1, min_width=90)
            clear = gr.Button("Xóa", scale=1, min_width=90)
        gr.Examples(EXAMPLES, inputs=q, label="Câu hỏi mẫu")
        info = gr.Markdown(elem_id="meta")
        with gr.Accordion("Đoạn tài liệu được dùng để trả lời", open=False):
            ctx = gr.Markdown()
        with gr.Accordion("Cài đặt", open=False):
            run = gr.Dropdown(ORDER, value=ORDER[0], label="Phương pháp truy hồi",
                              info="Mặc định là phương pháp cho kết quả tốt nhất trong benchmark.")
        for ev in (q.submit, send.click):
            ev(ask_rag, [q, run, chatbot], [chatbot, info, ctx]).then(lambda: "", None, q)
        clear.click(lambda: ([], "", ""), None, [chatbot, info, ctx])
    with gr.Tab("📄 Trích xuất PO"):
        build_po_tab()

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=int(os.environ.get("BIBICA_PORT", "7860")),
                        theme=THEME, css=CSS)
