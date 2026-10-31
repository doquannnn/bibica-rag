"""Simple UI: ask questions over the Bibica HR documents, or extract retailer purchase orders (PO PDF -> the fields
that retailer needs, see bibica_rag/po).

    scripts/serve.sh 4B            # Qwen3.5-4B on :8080 (default model)
    scripts/serve.sh 9B 8082       # optional: Qwen3.5-9B on :8082, selectable in the "Model" dropdown
    # PO extraction (provider "local") uses the selected model; optional: BIBICA_PO_PROVIDER=gemini GEMINI_API_KEY=...
    uv run python app.py           # http://127.0.0.1:7860 (other port: BIBICA_PORT=7861)
Other model/endpoint pairs: LLM_ENDPOINTS="qwen3.5-4B=http://127.0.0.1:8080/v1,qwen3.5-9B=http://127.0.0.1:8082/v1".
"""
import json
import os
import time
from functools import lru_cache

import gradio as gr
import yaml

from bibica_rag.corpus import ROOT
from bibica_rag.eval.run_pipelines import build_pipeline
from bibica_rag.llm import ENDPOINTS, MODEL, server_status, use_model
from bibica_rag.po.ui import CSS, build_po_tab

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


MODELS = list(ENDPOINTS)


def models_status_md():
    """One short line: a dot per model (the endpoint is in the error shown when an offline model is used)."""
    return "**Trạng thái:** " + " · ".join(
        f"🟢 {m}" if (st := server_status(m)) == "online" else f"🔴 {m} ({st})" for m in MODELS)


def check_model(model):
    st = server_status(model)
    if st != "online":
        raise gr.Error(f"Model {model} chưa sẵn sàng tại {ENDPOINTS[model]}: {st}. "
                       f"Chạy: scripts/serve.sh {model.split('-')[-1]} {ENDPOINTS[model].split(':')[-1].split('/')[0]}")


@lru_cache(maxsize=None)
def get_pipe(run, model):
    with use_model(model):  # framework pipelines (DSPy, LangChain, ...) keep the model they were built with
        return build_pipeline(SPECS[run])


def ask_rag(question, run, model, history):
    history = history or []
    if not question.strip():
        return history, "", ""
    check_model(model)
    t0 = time.time()
    with use_model(model):
        r = get_pipe(run, model).answer(question)
    history = history + [{"role": "user", "content": question}, {"role": "assistant", "content": r["answer"]}]
    t = r["timings"]
    info = (f"**Pipeline:** `{run}` · **Model:** `{model}`  \n"
            f"**Nguồn truy hồi:** {', '.join(r.get('retrieved_nodes', [])[:8]) or '(toàn văn)'}  \n"
            f"**Thời gian:** truy hồi {t.get('retrieval_s') or 0:.1f}s · token đầu {t.get('ttft_s') or 0:.1f}s · "
            f"tổng {time.time() - t0:.1f}s · {r.get('llm_calls', 1)} lần gọi LLM")
    ctx = "\n\n---\n\n".join(f"**{', '.join(c['node_ids'])}**\n\n{c['context']}" for c in r.get("contexts", [])[:8])
    return history, info, ctx or "(pipeline không truy hồi đoạn văn — dùng toàn văn / wiki)"


with gr.Blocks(title="Bibica") as demo:
    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            gr.Markdown("## Bibica — Trợ lý tài liệu nội bộ\n"
                        "Hỏi đáp Quy chế lương thưởng · Nội quy lao động · Thỏa ước LĐTT, và trích xuất đơn đặt hàng (PO).")
        with gr.Column(scale=2, min_width=280):
            model = gr.Dropdown(MODELS, value=MODEL, label="Model (9B chính xác hơn, chậm hơn ~1,5×)")
            with gr.Row(equal_height=True):
                status = gr.Markdown(models_status_md(), elem_classes="status")
                refresh = gr.Button("Kiểm tra lại", size="sm", scale=0, min_width=110)
    refresh.click(models_status_md, None, status)
    model.change(models_status_md, None, status)

    with gr.Tab("Hỏi đáp"):
        chatbot = gr.Chatbot(height=440, label="Hội thoại")
        with gr.Row(equal_height=True):
            q = gr.Textbox(placeholder="Ví dụ: Làm thêm giờ vào ngày lễ được trả bao nhiêu phần trăm lương?",
                           show_label=False, lines=2, scale=5, container=False)
            ask = gr.Button("Hỏi", variant="primary", scale=1, min_width=100)
        info = gr.Markdown()
        with gr.Accordion("Đoạn tài liệu đã dùng để trả lời", open=False):
            ctx = gr.Markdown()
        with gr.Accordion("Cấu hình", open=False):
            run = gr.Dropdown(ORDER, value=ORDER[0], label="Pipeline (mặc định: tốt nhất theo benchmark)")
        for ev in (q.submit, ask.click):
            ev(ask_rag, [q, run, model, chatbot], [chatbot, info, ctx]).then(lambda: "", None, q)
    with gr.Tab("Trích xuất PO"):
        build_po_tab(local_model=model)
    demo.load(models_status_md, None, status)

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=int(os.environ.get("BIBICA_PORT", "7860")), css=CSS)
