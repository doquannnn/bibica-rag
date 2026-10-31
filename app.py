"""Simple UI: ask questions over the Bibica HR documents, extract retailer purchase orders (PO PDF -> the fields
that retailer needs, see bibica_rag/po), or extract information from an image with the Qwen VLM.

    scripts/serve.sh 4B            # llama-server on :8080 (or 9B)
    export GEMINI_API_KEY=...      # optional: PO extraction with Gemini instead of the local VLM
    uv run python app.py           # http://127.0.0.1:7860
"""
import json
import time
from functools import lru_cache

import gradio as gr
import yaml

from bibica_rag.corpus import ROOT
from bibica_rag.eval.run_pipelines import build_pipeline
from bibica_rag.llm import MODEL, chat, image_part
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
    info = (f"**Pipeline:** `{run}` · **Model:** `{MODEL}`  \n"
            f"**Nguồn truy hồi:** {', '.join(r.get('retrieved_nodes', [])[:8]) or '(toàn văn)'}  \n"
            f"**Thời gian:** truy hồi {t.get('retrieval_s') or 0:.1f}s · token đầu {t.get('ttft_s') or 0:.1f}s · "
            f"tổng {time.time() - t0:.1f}s · {r.get('llm_calls', 1)} lần gọi LLM")
    ctx = "\n\n---\n\n".join(f"**{', '.join(c['node_ids'])}**\n\n{c['context']}" for c in r.get("contexts", [])[:8])
    return history, info, ctx or "(pipeline không truy hồi đoạn văn — dùng toàn văn / wiki)"


DEFAULT_IMG_PROMPT = ("Đây là ảnh chụp một chứng từ (ví dụ đơn đặt hàng). Hãy trích xuất toàn bộ thông tin dưới dạng JSON "
                      "với các trường: so_don_hang, ngay, ben_mua, ben_ban, dia_chi_giao, danh_sach_hang (ten_hang, "
                      "don_vi, so_luong, don_gia, thanh_tien), tong_tien, ghi_chu. Trường nào không có thì để null. "
                      "Chỉ in ra JSON.")


def ask_image(img_path, prompt):
    if not img_path:
        return "Hãy tải ảnh lên."
    t0 = time.time()
    r = chat([{"role": "user", "content": [image_part(img_path), {"type": "text", "text": prompt}]}], max_tokens=1500)
    return f"{r['text']}\n\n---\n_{MODEL} · {time.time() - t0:.1f}s_"


with gr.Blocks(title="Bibica RAG") as demo:
    gr.Markdown("## Bibica — Hỏi đáp tài liệu nội bộ (Quy chế lương thưởng · Nội quy lao động · Thỏa ước LĐTT) "
                "· Trích xuất PO")
    with gr.Tab("Hỏi đáp"):
        with gr.Row():
            run = gr.Dropdown(ORDER, value=ORDER[0], label="Pipeline (mặc định: tốt nhất theo benchmark)", scale=3)
        chatbot = gr.Chatbot(height=420)
        q = gr.Textbox(placeholder="Ví dụ: Làm thêm giờ vào ngày lễ được trả bao nhiêu phần trăm lương?",
                       label="Câu hỏi", lines=2)
        info = gr.Markdown()
        with gr.Accordion("Các đoạn tài liệu đã truy hồi", open=False):
            ctx = gr.Markdown()
        q.submit(ask_rag, [q, run, chatbot], [chatbot, info, ctx]).then(lambda: "", None, q)
        gr.Button("Hỏi", variant="primary").click(ask_rag, [q, run, chatbot], [chatbot, info, ctx]).then(
            lambda: "", None, q)
    with gr.Tab("Trích xuất PO"):
        build_po_tab()
    with gr.Tab("Đọc ảnh (VLM)"):
        with gr.Row():
            img = gr.Image(type="filepath", label="Ảnh chứng từ / đơn hàng")
            with gr.Column():
                prompt = gr.Textbox(DEFAULT_IMG_PROMPT, label="Yêu cầu", lines=5)
                out = gr.Markdown()
        gr.Button("Trích xuất", variant="primary").click(ask_image, [img, prompt], out)

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
