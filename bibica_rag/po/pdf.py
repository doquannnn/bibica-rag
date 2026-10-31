"""PDF -> page images (for the VLM) + text layer (hint for digital PDFs, empty for scans)."""
import base64
import io
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image


@dataclass
class PdfPages:
    images: list[Image.Image]
    texts: list[str]
    n_pages: int  # total pages in the file (images/texts may be truncated to max_pages)

    @property
    def text(self) -> str:
        return "\n\n".join(f"--- Trang {i + 1} ---\n{t}" for i, t in enumerate(self.texts) if t.strip())


def load_pdf(path: str | Path, dpi: int = 150, max_pages: int = 4, max_side: int = 1800) -> PdfPages:
    """Render the first `max_pages` pages. `max_side` caps the longest image side (VLM token budget)."""
    doc = pdfium.PdfDocument(str(path))
    try:
        images, texts = [], []
        for i in range(min(len(doc), max_pages)):
            page = doc[i]
            img = page.render(scale=dpi / 72).to_pil().convert("RGB")
            if max(img.size) > max_side:
                img.thumbnail((max_side, max_side), Image.LANCZOS)
            images.append(img)
            tp = page.get_textpage()
            texts.append(tp.get_text_range().replace("\r\n", "\n").replace("\r", "\n"))
            tp.close()
            page.close()
        return PdfPages(images=images, texts=texts, n_pages=len(doc))
    finally:
        doc.close()


def image_data_url(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"
