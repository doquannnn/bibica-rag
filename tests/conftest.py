import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeClient:
    """Stands in for openai.OpenAI: records requests, replies with queued strings (or raises queued errors)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if not isinstance(reply, str):
            reply = json.dumps(reply, ensure_ascii=False)
        if kwargs.get("stream"):
            return iter(SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=reply[i:i + 5]))])
                        for i in range(0, len(reply), 5))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


@pytest.fixture
def fake_client():
    return FakeClient


def make_pdf(path: Path, pages: list[list[str]]):
    """Tiny text PDF (ASCII only: the built-in font has no Vietnamese glyphs)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    for lines in pages:
        y = 800
        for line in lines:
            c.drawString(50, y, line)
            y -= 18
        c.showPage()
    c.save()
    return path


@pytest.fixture
def pdf_factory(tmp_path):
    def _make(name, pages):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return make_pdf(p, pages)
    return _make


def write_xlsx(path: Path, sheets: dict[str, list[list]]):
    from openpyxl import Workbook

    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(r)
    wb.save(path)
    return path


@pytest.fixture
def xlsx_factory(tmp_path):
    def _make(sheets, name="spec.xlsx"):
        return write_xlsx(tmp_path / name, sheets)
    return _make
