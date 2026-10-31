"""Vietnamese prompts shared by all pipelines (kept identical so pipelines differ only in retrieval/orchestration)."""

SYSTEM_QA = """Bạn là trợ lý nhân sự nội bộ của Công ty Cổ phần Bibica. Bạn trả lời câu hỏi của nhân viên CHỈ dựa trên các trích đoạn tài liệu nội bộ được cung cấp:
- QCL = Quy chế tiền lương, tiền thưởng Bibica năm 2024
- NQLD = Nội quy lao động 2024
- TULD = Thỏa ước lao động tập thể 2024

Quy tắc:
1. Chỉ dùng thông tin có trong trích đoạn. Không suy đoán, không dùng kiến thức bên ngoài.
2. Trả lời đầy đủ, chính xác số liệu (giữ nguyên con số, tỷ lệ %, số ngày, mức tiền như trong tài liệu).
3. Sau mỗi ý, ghi nguồn trong ngoặc vuông theo dạng [QCL Điều 12], [NQLD Điều 7], [QCL Phụ lục 2].
4. Nếu câu hỏi liên quan nhiều tài liệu, nêu rõ từng tài liệu quy định gì; nếu chúng khác nhau thì chỉ ra điểm khác.
5. Nếu trích đoạn không chứa thông tin để trả lời, chỉ trả lời: "Tài liệu nội bộ không có thông tin về vấn đề này." (có thể nêu ngắn gọn thông tin liên quan nếu có).
6. Trả lời bằng tiếng Việt, ngắn gọn (thường dưới 150 từ), đi thẳng vào vấn đề; không nhắc lại câu hỏi, không mở đầu bằng "Dựa trên các trích đoạn..."."""


def format_contexts(contexts):
    """contexts: list of dicts with node_ids + context text."""
    blocks = []
    for i, c in enumerate(contexts, 1):
        src = ", ".join(c.get("node_ids", [])) or "?"
        blocks.append(f"[Trích đoạn {i} | nguồn: {src}]\n{c['context']}")
    return "\n\n".join(blocks)


def qa_user(question, contexts):
    return f"TRÍCH ĐOẠN TÀI LIỆU:\n{format_contexts(contexts)}\n\nCÂU HỎI: {question}\n\nTRẢ LỜI:"


def qa_messages(question, contexts):
    return [{"role": "system", "content": SYSTEM_QA}, {"role": "user", "content": qa_user(question, contexts)}]


MULTI_QUERY = """Viết lại câu hỏi sau thành {n} câu truy vấn tìm kiếm khác nhau bằng tiếng Việt CÓ DẤU, đầy đủ, dùng thuật ngữ của văn bản quy chế/nội quy công ty (ví dụ: "người lao động", "tiền lương làm thêm giờ", "kỷ luật lao động"). Nếu câu hỏi viết không dấu hoặc sai chính tả, hãy sửa lại. Nếu câu hỏi gồm nhiều ý, mỗi truy vấn tập trung một ý.
Chỉ in ra mỗi truy vấn trên một dòng, không đánh số, không giải thích.

Câu hỏi: {q}"""

HYDE = """Hãy viết một đoạn văn ngắn (3-5 câu) theo văn phong của quy chế/nội quy lao động của một công ty Việt Nam, có thể là đoạn trả lời cho câu hỏi dưới đây. Viết tiếng Việt có dấu, không giải thích thêm.

Câu hỏi: {q}"""

CONTEXTUALIZE = """<document>
{doc}
</document>
Đây là một đoạn trích từ tài liệu trên:
<chunk>
{chunk}
</chunk>
Hãy viết 1-2 câu ngắn bằng tiếng Việt mô tả đoạn trích này nằm ở đâu trong tài liệu và nói về chủ đề gì, nhằm giúp tìm kiếm đoạn trích tốt hơn. Chỉ in ra câu mô tả."""

GRADE_DOCS = """Câu hỏi: {q}

Các trích đoạn đã tìm được:
{ctx}

Các trích đoạn trên có đủ thông tin để trả lời ĐẦY ĐỦ câu hỏi không? Trả lời JSON: {{"sufficient": true/false, "missing": "thông tin còn thiếu (nếu có)", "better_query": "câu truy vấn tìm kiếm tốt hơn, tiếng Việt có dấu"}}"""
