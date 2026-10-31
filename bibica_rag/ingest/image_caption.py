"""Caption the images embedded in the documents with the local Qwen VLM → data/parsed/image_captions.json.

The captions replace the `[HÌNH ẢNH: name]` placeholders when the corpus is loaded (see corpus.load_nodes).
"""
import json

from ..corpus import CAPTIONS, DATA
from ..llm import chat, image_part

PROMPT = ("Đây là một hình ảnh trong văn bản nội bộ của Công ty Cổ phần Bibica. Hãy chép lại CHÍNH XÁC toàn bộ chữ "
          "trong ảnh (giữ nguyên dấu, số, ngày giờ), sau đó mô tả ngắn gọn ảnh là gì (ví dụ: logo, chữ ký số...). "
          "Trả lời một đoạn tiếng Việt, không quá 3 câu.")


def main():
    caps = {}
    for p in sorted((DATA / "images").iterdir()):
        r = chat([{"role": "user", "content": [image_part(p), {"type": "text", "text": PROMPT}]}], max_tokens=200)
        caps[p.name] = " ".join(r["text"].split())
        print(f"{p.name} ({r['total_s']:.1f}s): {caps[p.name]}")
    CAPTIONS.write_text(json.dumps(caps, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
