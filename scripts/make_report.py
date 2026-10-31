"""results/leaderboard.json -> results/report.html (self-contained)."""
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
L = json.loads((ROOT / "results" / "leaderboard.json").read_text(encoding="utf-8"))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def f(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


judged = [r for r in L if r["n_judged"] >= 200]
rows = []
for r in judged:
    lo, hi = wilson(r["pass"], r["n"])
    lat = f"{r['latency_p50_s']:.1f} / {r['latency_p95_s']:.1f}"
    src = "tuần tự" if r["latency_src"] == "sequential" else "đồng thời ×4"
    rows.append(f"<tr><td>{r['rank']}</td><td><b>{html.escape(r['run'].split('@')[0])}</b></td>"
                f"<td>{html.escape(r['technique'])}</td><td>{html.escape(r['framework'])}</td>"
                f"<td>{html.escape(r['chunking'])}</td>"
                f"<td class=n><b>{r['pass']}/{r['n']}</b><br><small>{r['pass_rate']*100:.1f}% [{lo*100:.0f}–{hi*100:.0f}]</small></td>"
                f"<td class=n>{f(r['correctness'])}</td><td class=n>{f(r['faithfulness'])}</td>"
                f"<td class=n>{f(r['citation_ok'])}</td><td class=n>{f(r['keyfact_all'])}</td>"
                f"<td class=n>{f(r['refusal_acc'],2)}</td><td class=n>{f(r['recall@5'])}/{f(r['mrr@10'])}</td>"
                f"<td class=n>{lat}<br><small>{src}</small></td></tr>")
un = [r for r in L if r["n_judged"] < 200]
unrows = "".join(f"<tr><td><b>{html.escape(r['run'].split('@')[0])}</b></td><td>{html.escape(r['technique'])}</td>"
                 f"<td class=n>{r['n']}</td><td class=n>{f(r['keyfact_all'])}</td><td class=n>{f(r['recall@5'])}/{f(r['mrr@10'])}</td>"
                 f"<td class=n>{r['latency_p50_s']:.1f}</td></tr>" for r in un)

page = f"""<!doctype html><html lang=vi><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Bibica RAG benchmark</title>
<style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#6b6b70;--line:#e3e3e6;--acc:#0b6bcb;--hl:#eaf3fd}}
@media(prefers-color-scheme:dark){{:root{{--bg:#141416;--fg:#ececee;--mut:#9a9aa0;--line:#2c2c30;--acc:#6db2ff;--hl:#18283a}}}}
body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto}}h1{{font-size:26px;margin:0 0 4px}}h2{{font-size:18px;margin:32px 0 8px}}
.mut,small{{color:var(--mut)}}.tw{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{color:var(--mut);font-weight:600}}
td.n{{text-align:right;white-space:nowrap}}tr:nth-child(-n+4) td{{background:var(--hl)}}
ul{{padding-left:20px}}code{{background:var(--hl);padding:1px 4px;border-radius:3px}}
</style><main>
<h1>Bibica RAG — kết quả benchmark</h1>
<p class=mut>3 tài liệu nhân sự (Quy chế lương thưởng, Nội quy lao động, Thỏa ước lao động tập thể 2024) · 240 câu hỏi test · generator Qwen3.5-4B (Q4_K_M, llama.cpp Vulkan, tắt thinking, trả lời ≤400 token)</p>
<h2>Kết luận</h2><ul>
<li><b>Tốt nhất về số câu đạt:</b> <code>cag-full-context</code> (nhồi toàn bộ ~23,5k token tài liệu vào prompt) 175/240; đứng sau là <code>dspy</code> 172 và <code>naive-best-s2b</code> 171. Ba cấu hình này chênh nhau 4 câu, nằm trong khoảng sai số (khoảng tin cậy 95% rộng ~±6 điểm %), nên coi là <b>ngang nhau</b>.</li>
<li><b>Đề xuất triển khai:</b> <code>naive-best-s2b</code> (chunk theo khoản + trả về cả Điều cha, hybrid vn-embedding + Qwen3-Embedding + BM25 không dấu/RRF). Faithfulness cao nhất (0,833), chạy trên server chuẩn, độ trễ tuần tự ~9 s/câu. CAG cần server riêng với ngữ cảnh 50k/slot.</li>
<li>Giao diện (<code>app.py</code>) mặc định là <code>dspy</code> (pipeline tốt nhất chạy được trên server chuẩn :8080).</li>
<li>Reranker bge-reranker-v2-m3 nâng Recall@5 ở Stage A (0,961→0,976) nhưng <b>không</b> tăng số câu đạt end-to-end (154 so với 165) và tốn ~18 s CPU/câu. Query transform (HyDE, multi-query), RAPTOR, CRAG và wiki không vượt retrieval tốt + prompt trả lời tốt.</li>
<li>LLM-Wiki do Claude biên soạn 164/240 (chậm ~100 s/câu); wiki do Qwen 4B tự biên soạn chỉ 89/240.</li>
<li>Framework dùng cùng chunk và embedder: DSPy 172, LangChain 163, LlamaIndex 159, Haystack 153 so với bản tự viết <code>naive-qwen3hyb</code> 154. Chưa phân tích nguyên nhân chênh lệch.</li></ul>
<h2>Bảng xếp hạng (đã chấm, theo số câu đạt)</h2>
<div class=tw><table><tr><th>#<th>Pipeline<th>Kỹ thuật<th>Framework<th>Chunking<th>Đạt<br><small>% [CI95 Wilson]</small><th>Correct<th>Faithful<th>Citation<th>Key-fact đủ<th>Từ chối đúng<th>Recall@5/MRR<th>p50/p95 (s)</tr>
{''.join(rows)}</table></div>
<p class=mut>Hàng tô nền = top 4. Độ trễ của 4 pipeline đầu đo <b>tuần tự</b> (1 luồng, 30 câu đầu); các pipeline còn lại đo khi chạy đồng thời 4 luồng nên lớn hơn ~4–5 lần và <b>không so sánh trực tiếp</b> với 4 hàng đầu.</p>
<h2>Chưa chấm (đã sinh câu trả lời)</h2>
<div class=tw><table><tr><th>Pipeline<th>Kỹ thuật<th>Số câu<th>Key-fact đủ<th>Recall@5/MRR<th>p50 đồng thời (s)</tr>{unrows}</table></div>
<p class=mut>LightRAG (mix, đồ thị tri thức do Qwen 4B trích xuất) đã chạy đủ 240 câu nhưng không chấm theo quyết định tiết kiệm; key-fact và retrieval thấp hơn hẳn các pipeline còn lại. Một số ngữ cảnh KG vượt slot 8k nên bị cắt còn 12.000 ký tự.</p>
<h2>Tiêu chí đánh giá</h2><ul>
<li><b>Đạt (pass):</b> câu có đáp án → correctness = 1 <i>và</i> faithful <i>và</i> đủ mọi key-fact; câu không có trong tài liệu → correctness = 1 (từ chối đúng) <i>và</i> faithful.</li>
<li><b>Correctness</b> 0 / 0,5 / 1 và <b>faithful</b>, <b>citation_ok</b>: Claude chấm theo <code>results/judge/RUBRIC.md</code>. <b>Key-fact:</b> khớp số liệu/dữ kiện bằng code. <b>Recall@5, MRR@10:</b> trên node gold (Điều/Phụ lục).</li>
<li>Bộ test 300 câu (dev 60 / test 240), 10 loại câu hỏi, kiểm chứng chéo với văn bản gốc.</li>
<li>Độ tin cậy judge: chấm lại độc lập 40 câu, trùng pass/fail 39/40 (97,5%).</li></ul>
<h2>Lưu ý / hạn chế</h2><ul>
<li>Judge đổi mô hình giữa chừng: 1193 phán quyết đầu do Opus, phần sau do Sonnet 5.5 (medium). Chưa kiểm tra độ lệch giữa hai judge.</li>
<li>Chỉ chạy Qwen3.5-4B; <b>chưa chạy Stage C với Qwen3.5-9B</b> và biến thể bật thinking (bỏ để tiết kiệm). Xếp hạng có thể đổi với model lớn hơn.</li>
<li>Thỏa ước tập thể được ký nhân danh Pan chứ không phải Bibica; bộ test và judge đã tính đến điểm này.</li>
<li>Nhiều pipeline thất bại cùng một kiểu: sai số học bảng KPI/lương (QCL-053/055/059/012), từ chối nhầm khi tài liệu có đáp án (NQLD-083, QCL-052).</li>
<li>MULTI-025 được chấm lại (16 câu trả lời) sau khi rubric được sửa.</li></ul>
</main></html>"""
(ROOT / "results" / "report.html").write_text(page, encoding="utf-8")
print("ok", len(page))
