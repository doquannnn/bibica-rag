"""Synthetic test POs: 3 per retailer, in the layout of that retailer's real PO, with ground truth.

Purpose: check that the extractor generalizes beyond the 9 real samples the field rules were written from —
other store codes, dates (ambiguous day/month, month/year boundaries), line counts (1 to ~45 lines, several
pages), number formats, scans. The layouts copy the real ones, so this does NOT test new layouts: real POs remain
the reference. Every page carries a footer saying the document is synthetic.

    uv run python scripts/make_synthetic_po.py         # -> data/po_synth/pdf/*.pdf + data/po_synth/gt/*.json

Deterministic (fixed seed). Ground truth follows config/po_customers.yaml: printed delivery date (+5 days for
BigC / Satra / Lan Chi order dates), store code as printed, product code + ordered quantity per line.
"""
import io
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

from PIL import Image, ImageFilter
from reportlab.lib.colors import Color, black, white
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bibica_rag.po.validate import ean13_ok  # noqa: E402

OUT = ROOT / "data" / "po_synth"
FONTS = {"R": "DejaVuSans.ttf", "B": "DejaVuSans-Bold.ttf", "M": "DejaVuSansMono.ttf", "S": "DejaVuSerif.ttf",
         "SB": "DejaVuSerif-Bold.ttf"}
for k, f in FONTS.items():
    pdfmetrics.registerFont(TTFont(k, f"/usr/share/fonts/truetype/dejavu/{f}"))
FOOTER = "DỮ LIỆU TỔNG HỢP ĐỂ KIỂM THỬ — KHÔNG PHẢI ĐƠN HÀNG THẬT"
GREY = Color(0.82, 0.82, 0.82)

# Bibica products (barcode, name) taken from the real POs; all valid EAN-13
PRODUCTS = [
    ("8934609203093", "BANH JOKER DAU 95G"), ("8934609203116", "BANH JOKER CA PHE 95G"),
    ("8934609202478", "BANH CHOCO CHIPS DIEU 144G"), ("8934609602322", "BANH HURA DELI COM DUA 336G"),
    ("8934609602377", "BANH HURA DELI BO SUA 168G"), ("8934609602452", "BANH SWISSROLL HURA COM 360G"),
    ("8934609602551", "BANH SWISSROLL HURA CAM 360G"), ("8934609602698", "BANH HURA HUONG COM 300G"),
    ("8934609602704", "BANH HURA BO SUA 300G"), ("8934609102532", "KEO NHAN TC TU QUY 400GR"),
    ("8934609106301", "KEO CUNG WELLY CAM GOI 70G"), ("8934609106318", "KEO CUNG WELLY DAU GOI 70G"),
    ("8934609106332", "KEO CUNG WELLY OI GOI 70G"), ("8934609411337", "KEO CUNG TRAI CAY 4 MUA 70G"),
    ("8934609412136", "KEO CUNG MIGITA ME 70G"), ("8934609413133", "KEO CUNG MIGITA BAC HA 70G"),
    ("8934609418336", "KEO CUNG MIGITA GUNG 70G"), ("8934609101900", "KEO MEM SOCOLA SUA 70G"),
    ("8934609101917", "KEO MEM SUA SUMIKA 70G"), ("8934609103164", "KEO DEO ZOO 200G"),
    ("8934609103188", "KEO DEO ZOO JELLY 100G"), ("8934609103195", "KEO DEO AO DUONG ZOO 500G"),
    ("8934609103218", "KEO DEO AO DUONG ZOO 200G"), ("8934609103706", "KEO DEO BIBICA HU 300G"),
    ("8934609103881", "KEO DEO ZOO SAU TUYET 96G"), ("8934609103911", "KEO DEO ZOO MONSTER VI SUA 50G"),
    ("8934609103928", "KEO DEO ZOO MONSTER VI SUA 96G"), ("8934609103997", "KEO DEO LOI KHUAN HURO HU 168G"),
    ("8934609105434", "KEO MEM SUA SUMIKA 275G"), ("8934609105533", "KEO MEM SUA AHHA TUI 60G"),
    ("8934609105540", "KEO MEM SUA AHHA TUI 120G"), ("8934609105601", "KEO MEM SUMIKA SOCOLA SUA 275G"),
    ("8934609105830", "KEO MEM AHHA TRAN CHAU DAU 50G"), ("8934609105847", "KEO MEM AHHA TRAN CHAU DAU 320G"),
    ("8934609105861", "KEO MEM AHHA TRAN CHAU CAM 50G"), ("8934609105939", "KEO MEM CHEERY DAU GOI 70G"),
    ("8934609105953", "KEO MEM CHEERY CAM GOI 70G"), ("8934609105991", "KEO MEM CHEERY NHO GOI 70G"),
    ("8934609109012", "KEO DEO LOI KHUAN HURO GOI 105G"), ("8934609471355", "KEO DEO ZOO 500G"),
    ("8934609202461", "BANH GOODY CHIP HAT DIEU TUI 80G"), ("8934609202492", "BANH GOODY CHIP DUA TUI 80G"),
    ("8934609202522", "BANH GOODY CHIP NHO TUI 80G"), ("8934609205837", "BANH BONG LAN QUASURE LIGHT COM 126G"),
    ("8934609602407", "BANH HURA DELI SUA DAU 336G"), ("8934609602605", "BANH HURA SWISSROLL KEM BO SUA 360G"),
    ("8934609602506", "BANH HURA SWISSROLL DAU 360G"), ("8934609605330", "BANH HURA SWISSROLL COM 180G"),
    ("8934609102143", "KEO TRAI CAY BON MUA 650G"), ("8934609105175", "KEO MEM SUMIKA SUA 140G"),
]
assert all(ean13_ok(c) for c, _ in PRODUCTS)
rng = random.Random(20261031)


def pick(n):
    return rng.sample(PRODUCTS, n)


def fmt_int(x, sep=","):
    return f"{x:,}".replace(",", sep)


def fmt_dec(x, nd=2, thou=",", dec="."):
    s = f"{x:,.{nd}f}"
    return s.replace(",", "\0").replace(".", dec).replace("\0", thou)


class Doc:
    """reportlab canvas with small helpers; y grows downward from the top margin."""

    def __init__(self, path, size=A4):
        self.c = canvas.Canvas(str(path), pagesize=size)
        self.W, self.H = size
        self.page = 0

    def y(self, top):
        return self.H - top

    def text(self, x, top, s, size=9, font="R", align="l", color=black):
        c = self.c
        c.setFillColor(color)
        c.setFont(font, size)
        s = str(s)
        if align == "r":
            c.drawRightString(x, self.y(top), s)
        elif align == "c":
            c.drawCentredString(x, self.y(top), s)
        else:
            c.drawString(x, self.y(top), s)
        c.setFillColor(black)

    def rect(self, x, top, w, h, fill=None, stroke=True, lw=0.6):
        c = self.c
        c.setLineWidth(lw)
        if fill is not None:
            c.setFillColor(fill)
        c.rect(x, self.y(top + h), w, h, stroke=int(stroke), fill=int(fill is not None))
        c.setFillColor(black)

    def line(self, x1, t1, x2, t2, lw=0.6):
        self.c.setLineWidth(lw)
        self.c.line(x1, self.y(t1), x2, self.y(t2))

    def grid_row(self, x, top, cols, cells, h, size=8, font="R", fill=None, border=True):
        """cols: [(width, align)], cells: [str or [str, ...]] (several lines)."""
        cx = x
        for (w, al), cell in zip(cols, cells):
            if border:
                self.rect(cx, top, w, h, fill=fill)
            lines = cell if isinstance(cell, list) else [cell]
            for k, s in enumerate(lines):
                ty = top + 3 + size + k * (size + 2)
                tx = cx + 3 if al == "l" else cx + w - 3 if al == "r" else cx + w / 2
                self.text(tx, ty, s, size, font, al)
            cx += w

    def end_page(self):
        self.text(self.W / 2, self.H - 12, FOOTER, 6.5, "R", "c", Color(0.45, 0.45, 0.45))
        self.c.showPage()
        self.page += 1

    def save(self):
        self.c.save()


def chunks(rows, n):
    return [rows[i:i + n] for i in range(0, len(rows), n)] or [[]]


# ---------------------------------------------------------------- AEON (scanned: image only)

def aeon(path, po, store, store_name, order_d, deliv_d, items):
    d = Doc(path, landscape(A4))
    pages = chunks(items, 11)
    for pi, part in enumerate(pages):
        d.text(d.W - 20, 22, f"trang {pi + 1} của {len(pages)}", 8, align="r")
        d.rect(30, 28, 210, 32, lw=1)
        d.text(135, 50, "STORE ORDER", 15, "B", "c")
        d.text(330, 40, "ÆON  AEON Vietnam Co, Ltd", 13, "B")
        d.text(330, 60, "ĐƠN ĐẶT HÀNG PHÂN PHỐI VỀ CỬA HÀNG", 9)
        d.rect(d.W - 150, 30, 130, 34)
        d.text(d.W - 146, 41, "Số đơn hàng", 7.5)
        d.text(d.W - 146, 58, po, 12)
        d.text(30, 75, "TẠO BỞI : 1002  CANARY", 7)
        d.rect(500, 58, 75, 26)
        d.text(537, 67, "VAT %", 7, align="c")
        d.text(537, 80, "8.00", 7, align="c")
        d.text(600, 72, "Ngày In", 7)
        d.text(600, 82, f"{order_d.day + 1:02d}   {order_d.month:02d}   {order_d.year}", 7)
        hcols = [(55, "c"), (110, "l"), (60, "c"), (150, "l"), (60, "c"), (55, "c"), (110, "c"), (60, "c"), (60, "c")]
        d.grid_row(30, 95, hcols, ["Loại Phiếu", "Được Chấp Thuận Bởi", "Mã Cửa Hàng", "Tên cửa hàng", "Mã Ngành Hàng",
                                   "Mã Bộ Phận", "Để giao hàng", "Ngày Đặt Hàng", "Ngày Giao Hàng"], 12, 6.5)
        d.grid_row(30, 107, hcols, ["321", ["268553/NGUYEN QUOC", "THANH"], store, store_name, "2", "2002",
                                    f"DC {store} RDCXD", order_d.strftime("%d   %m   %Y"),
                                    deliv_d.strftime("%d   %m   %Y")], 24, 6.5)
        cols = [(22, "c"), (240, "l"), (55, "r"), (60, "c"), (65, "c"), (85, "r"), (45, "r"), (50, "c"), (45, "r"),
                (55, "r"), (60, "r")]
        d.grid_row(30, 140, cols, ["STT", "TêN HàNG", ["SL/KHI", "KíCH THướC"], ["ĐơN Vị TíNH", "MàU"],
                                   ["Mã HàNG", "Mã HàNG NCC"], "Mã VạCH", "SL ĐặT", ["SỐ LƯỢNG", "GIAO HÀNG"],
                                   ["CHIếT KHấU", "(%)"], "ĐơN Giá", "THàNH TIềN"], 22, 6.3)
        top = 162
        for k, it in enumerate(part):
            n = pi * 11 + k + 1
            d.grid_row(30, top, cols, [str(n), it["name"], f"{it['pack']:.2f}", f"CARTON {it['pack']}",
                                       it["ncc"], it["code"], f"{it['qty']:.2f}", "", "0.00",
                                       fmt_dec(it["price"], 3, ".", ","), fmt_dec(it["price"] * it["qty"], 3, ".", ",")],
                       26, 7)
            top += 26
        if pi == len(pages) - 1:
            total = sum(i["price"] * i["qty"] for i in items)
            d.text(d.W - 90, top + 14, "TổNG TIềN", 6.5, align="r")
            d.text(d.W - 22, top + 14, fmt_dec(total, 3, ".", ","), 7, align="r")
            d.text(d.W / 2, top + 60, "XáC NHậN VIệC GIAO HàNG", 7, align="c")
        d.end_page()
    d.save()


# ---------------------------------------------------------------- BHX

def bhx(path, po, kho, kho_name, order_d, d1, d2, items, old_style=False):
    d = Doc(path, landscape(A4))
    pages = chunks(items, 7)
    for pi, part in enumerate(pages):
        if pi == 0:
            d.rect(30, 22, 170, 42, fill=Color(0.1, 0.5, 0.2), stroke=False)
            d.text(115, 49, "Bách hóa XANH", 15, "B", "c", white)
            d.text(230, 38, "Mã đơn hàng:", 9, "B")
            d.text(320, 38, po, 9)
            d.text(230, 56, "Ngày đặt hàng:", 9, "B")
            d.text(320, 56, order_d.strftime("%d/%m/%Y") + "  08:12", 9, "B")
            d.text(470, 32, "CÔNG TY CỔ PHẦN THƯƠNG MẠI BÁCH HÓA XANH", 9, "B")
            d.text(470, 46, "VP: Lầu 5, Lô T2-1.2, Đường D1, Khu Công Nghệ Cao, P.Tân Phú, TP.HCM", 7)
            d.text(d.W / 2, 90, "ĐƠN ĐẶT HÀNG", 13, "B", "c")
            d.text(30, 112, "Tên chi nhánh NCC:", 8, "B")
            d.text(130, 112, "18141 - BHX <> CÔNG TY CỔ PHẦN PHÂN PHỐI HÀNG TIÊU DÙNG PAN", 8)
            d.text(30, 128, "Kho NCC giao hàng:", 8, "B")
            d.text(130, 128, f"{kho} - {kho_name}", 8)
            d.text(30, 144, "Mã số thuế:", 8, "B")
            d.text(130, 144, "1101912597", 8)
            d.text(470, 112, "Tên NCC:", 8, "B")
            d.text(560, 112, "36521 - CÔNG TY CỔ PHẦN PHÂN PHỐI HÀNG TIÊU DÙNG PAN", 8)
            d.text(470, 144, "Ngày NCC giao hàng:", 8, "B")
            d.text(570, 144, f"{d1:%d/%m/%Y} - {d2:%d/%m/%Y}", 9, "B")
            d.text(30, 162, "Căn cứ vào hợp đồng mua bán đã được hai bên thỏa thuận, chúng tôi đặt mua sản phẩm "
                            "theo chi tiết bên dưới.", 7.5)
            top = 170
        else:
            top = 30
        cols = [(28, "c"), (30, "c"), (55, "c"), (88, "c"), (160, "l"), (40, "c"), (55, "r"), (62, "r"), (62, "r"),
                (35, "c"), (75, "r"), (100, "l")]
        d.grid_row(25, top, cols, ["STT", "Loại", "Barcode", ["Mã sản phẩm/", "Barcode"], "Tên sản phẩm", "ĐVT",
                                   "SL", "Đơn giá", ["Đơn giá", "chiết khấu"], ["VAT", "(%)"], "Thành tiền",
                                   "Ghi chú"], 24, 7, "B")
        top += 24
        for k, it in enumerate(part):
            n = pi * 7 + k + 1
            top_code = it["internal"] if old_style else it["code"]
            note = [f"{it['qty'] // it['pack']} Thùng - (Thùng {it['pack']} {it['unit']})", it["carton"]]
            d.grid_row(25, top, cols, [str(n), "A", "", [top_code, "", it["code"]], [it["name"]], it["unit"],
                                       fmt_dec(it["qty"]), fmt_dec(it["price"], 2), fmt_dec(it["price"], 2), "8",
                                       fmt_dec(it["price"] * it["qty"], 2), note], 40, 7)
            top += 40
        if pi == len(pages) - 1:
            tq = sum(i["qty"] for i in items)
            d.grid_row(25, top, [(401, "l"), (55, "r"), (334, "l")], ["Tổng", fmt_int(tq), ""], 16, 7.5, "B")
            d.text(600, top + 40, "Tổng tiền ĐH:", 8)
            d.text(780, top + 40, fmt_int(round(sum(i["price"] * i["qty"] for i in items))), 8, align="r")
        d.end_page()
    d.save()


# ---------------------------------------------------------------- BigC

def bigc(path, po, contract, deliv_name, order_d, store_d, items, stores):
    d = Doc(path)
    pn = chunks(items, 30)
    n_pages = len(pn) + len(stores)
    for pi, part in enumerate(pn):
        d.text(25, 25, "NHA CUNG CAP LUU Y GIA TREN DON DAT HANG LA GIA MUA CHINH THUC", 7)
        d.text(25, 35, "NEU CO SAI BIET PHAI DIEU CHINH VOI TRUNG TAM THU MUA TRUOC KHI GIAO HANG", 7)
        d.text(d.W - 25, 35, f"Page {pi + 1} of {n_pages}", 8, "B", "r")
        d.text(d.W / 2, 62, "PURCHASE NOTE", 14, "B", "c")
        hc = [(95, "c"), (90, "c"), (80, "c"), (85, "c"), (55, "c"), (90, "c")]
        d.grid_row(40, 75, hc, ["Order No", "Order Date", "Supplier Code", "Com. Contract", "Ad. Ch.", "Order Type"],
                   16, 7.5, "B", fill=GREY)
        d.grid_row(40, 91, hc, [po, order_d.strftime("%d/%m/%y") + " 15:01", "3004613", contract, "1",
                                "Cross dock order"], 16, 7.5)
        bc = [(135, "l")] * 4
        d.grid_row(25, 118, bc, ["Ordered By", "Delivered To", "For Store", "By Supplier"], 13, 7, "B", fill=GREY)
        d.grid_row(25, 131, bc, [["CTY TNHH DICH VU EB", "163 Phan Dang Luu", "Phuong 1, Quan Phu Nhuan"],
                                 [deliv_name, "KCN VSIP 2, Tan Binh", "Tan Uyen, Binh Duong"],
                                 ["CTY TNHH DICH VU EB", "163 Phan Dang Luu", "Ho Chi Minh"],
                                 ["CTY CP PP HANG TIEU DUNG PAN", "Lô A1/1-9, Đ: VL3; KCN Vĩnh Lộc 2", "Long An"]],
                   44, 6.5)
        cols = [(72, "l"), (170, "l"), (35, "c"), (22, "c"), (35, "r"), (35, "r"), (35, "r"), (50, "r"), (25, "c"),
                (66, "r"), (25, "c")]
        top = 190
        d.grid_row(25, top, cols, ["Article", "Article Description", ["OU", "Type"], "LV", ["SKU/", "OU"], "OU Qty",
                                   "Free Qty", ["Net", "Purch.Price"], "Unit", ["Total Net", "Purch. Price"], "SO"],
                   22, 6.5, "B", fill=GREY)
        top += 22
        for it in part:
            d.grid_row(25, top, cols, [it["code"], it["name"][:34], "Pack", str(it["lv"]), str(it["pack"]),
                                       str(it["qty"]), "0", fmt_int(it["price"]), "Cai",
                                       fmt_int(it["price"] * it["pack"] * it["qty"]), ""], 18, 6.8)
            top += 18
        if pi == len(pn) - 1:
            tq = sum(i["qty"] for i in items)
            tv = sum(i["price"] * i["pack"] * i["qty"] for i in items)
            d.text(250, top + 14, "SWEET GROCERY (410)", 7.5, "B")
            d.text(392, top + 14, str(tq), 7.5, "B", "r")
            d.text(570, top + 14, fmt_int(tv), 7.5, "B", "r")
            d.text(40, top + 40, f"TOTAL BF.TAX   :   {fmt_int(tv)}", 7.5)
            d.text(40, top + 70, "Delivery Date To Store", 7.5, "B")
            d.text(250, top + 70, "Total Qty", 7.5, "B")
            d.text(40, top + 84, store_d.strftime("%d/%m/%y"), 7.5)
            d.text(250, top + 84, str(tq), 7.5)
        d.end_page()
    # store split pages: must NOT be extracted
    for si, (sname, split) in enumerate(stores):
        d.text(d.W - 25, 30, f"Page {len(pn) + si + 1} of {n_pages}", 8, "B", "r")
        d.text(d.W / 2, 55, "DETAIL DELIVERY BY STORE", 13, "B", "c")
        d.text(25, 75, f"SO number: {int(po) - 3000 - si}    Order date: {order_d:%d/%m/%y} 11:37:00    "
                       f"Delivery date: {store_d + timedelta(days=12):%d/%m/%y}", 7.5)
        d.text(25, 95, f"Delivered To: {deliv_name}      For Store: {sname}", 7.5)
        cols = [(90, "l"), (210, "l"), (60, "c"), (40, "c"), (60, "r"), (60, "r")]
        top = 115
        d.grid_row(25, top, cols, ["Article", "Article desc.", "OU Type", "LV", "SKU/OU", "OU Qty"], 14, 7, "B",
                   border=False)
        top += 16
        for it, q in split:
            d.grid_row(25, top, cols, [it["code"], it["name"][:38], "Pack", str(it["lv"]), str(it["pack"]), str(q)],
                       16, 7, border=False)
            top += 16
        d.end_page()
    d.save()


# ---------------------------------------------------------------- COOP (JDA text report, monospace)

def coop(path, po, loc, entry_d, exp_d, items):
    d = Doc(path, landscape(A4))
    pages = chunks(items, 26)
    for pi, part in enumerate(pages):
        m = lambda x, t, s: d.text(x, t, s, 7.2, "M")  # noqa: E731
        m(25, 30, "POM343")
        m(25, 39, "PDTUSRDTTL")
        m(25, 48, f"P/O Number:    {po}")
        m(25, 57, f"P/O Location:    {loc}")
        m(300, 39, "JDA Software Version 7.7.0 (PDN_DC)")
        m(330, 48, "Purchase Order")
        m(700, 30, f"Date: {entry_d:%d/%m/%y}")
        m(700, 39, f"Page:        {pi + 1}")
        m(25, 66, "Bill To:                          Vendor:    20230")
        m(25, 75, "Head Office SGC - 1                         110-Cty CP Phan Phoi HTD PAN")
        m(430, 75, "Buyer   - LB3 Dang T T Linh")
        m(640, 75, f"Entry Date           - {entry_d.day}/{entry_d.month:02d}/{entry_d:%y}")
        m(640, 84, f"Expected Receipt Date- {exp_d.day}/{exp_d.month:02d}/{exp_d:%y}")
        cancel = exp_d + timedelta(days=29)
        m(640, 93, f"Cancel Date          - {cancel.day}/{cancel.month:02d}/{cancel:%y}")
        m(640, 102, "Recvd Date           - 0/00/00")
        m(25, 111, "Ship To:")
        m(25, 120, "Trung Tam Phan phoi Song Than")
        m(25, 150, f"{'':66}{'Sell Buy':10}{'Buy':>12}{'Net Buy':>12}{'Qty Ord/':>10}{'Qty Ord/':>10}"
                   f"{'Qty Rec/':>10}{'Extended':>20}")
        m(25, 159, f"{'SKU Number   Description':45}{'Vendor Part No.':21}{'U/M U/M':10}{'Cost':>12}{'Cost':>12}"
                   f"{'CS':>10}{'Pcs':>10}{'Pcs':>10}{'Cost':>20}")
        top = 175
        for it in part:
            pcs = it["qty"] * it["pack"]
            cost = it["price"] * it["pack"]
            m(25, top, f"{it['sku']:13}{it['short'][:30]:32}{it['vpn']:21}{'EA':4}{'C' + str(it['pack']):6}"
                       f"{cost:>12.2f}{cost:>12.2f}{it['qty']:>10.2f}{pcs:>10.2f}{'.00':>10}"
                       f"{fmt_dec(cost * it['qty']):>20}")
            top += 10
        if pi == len(pages) - 1:
            tq = sum(i["qty"] for i in items)
            tp = sum(i["qty"] * i["pack"] for i in items)
            tv = sum(i["price"] * i["pack"] * i["qty"] for i in items)
            m(25, top + 12, f"{'':95}Sub Total -{tq:>10.2f}{tp:>10.2f}{'.00':>10}{fmt_dec(tv):>20}")
            m(25, top + 30, "              Notes - Xin vui long kem DDH khi giao hang")
            m(25, top + 48, f"{'':125}Total - {fmt_dec(tv):>20}")
        d.end_page()
    d.save()


# ---------------------------------------------------------------- E-Mart

def emart(path, po, deliv, order_d, deliv_d, items):
    d = Doc(path)
    d.text(d.W / 2, 40, "THISO RETAIL COMPANY LIMITED", 11, "B", "c")
    d.text(d.W / 2, 56, "Purchase Order", 11, "B", "c")
    d.text(d.W / 2, 72, "(PO Type : Standard PO)", 10, "B", "c")
    d.text(25, 100, "Vendor       :  CÔNG TY CỔ PHẦN PHÂN PHỐI HÀNG TIÊU DÙNG PAN (101348)", 7.5)
    d.text(25, 114, "PO No.", 7.5, "B")
    d.text(85, 114, f":  {po}", 7.5)
    d.text(25, 128, "Delivery to", 7.5, "B")
    d.text(85, 128, f":  {deliv}", 7.5)
    d.text(25, 142, "VAT", 7.5, "B")
    d.text(85, 142, ":  8%", 7.5)
    d.text(340, 114, "Order By / Date", 7.5, "B")
    d.text(420, 114, f":  {order_d:%d.%m.%Y} / FMCG Sweet", 7.5)
    d.text(340, 128, "Delivery Date", 7.5, "B")
    d.text(420, 128, f":  {deliv_d:%d.%m.%Y}", 7.5)
    d.text(340, 142, "Department", 7.5, "B")
    d.text(420, 142, ":  SWEET GROCERY (202000000)", 7.5)
    d.text(d.W - 25, 160, "Page : 1/1", 7.5, "B", "r")
    cols = [(52, "c"), (70, "c"), (170, "l"), (30, "c"), (35, "r"), (40, "r"), (55, "r"), (60, "r"), (33, "c")]
    yellow = Color(1, 0.93, 0.2)
    d.grid_row(25, 166, cols, ["Article Code", ["Unit Barcode", "(Mã vạch lẻ)"], ["Unit Barcode Description",
                               "(Tên sản phẩm lẻ)"], ["PO", "Unit"], ["Qty.", "inBox"], "PO Qty.", ["Pur. Price",
                               "(-VAT)"], ["Amount", "(-VAT)"], ["Free", "PO"]], 24, 6.8, "B", fill=yellow)
    top = 190
    for it in items:
        d.grid_row(25, top, cols, [it["article"], it["code"], it["name"][:36], "BOX", str(it["pack"]), str(it["qty"]),
                                   fmt_int(it["price"], "."), fmt_int(it["price"] * it["qty"], "."), ""], 14, 6.8)
        top += 14
    tv = sum(i["price"] * i["qty"] for i in items)
    d.text(330, top + 16, "Total Amount(without VAT) :", 7.5, "B")
    d.text(d.W - 30, top + 16, fmt_int(tv, "."), 7.5, align="r")
    d.text(25, top + 60, "Lưu ý / Attention : Vui lòng kiểm tra giá kỹ trước khi giao hàng.", 7)
    d.end_page()
    d.save()


# ---------------------------------------------------------------- Lotte

def lotte(path, po, store, city, order_d, sply_d, items):
    d = Doc(path, landscape(A4))
    pages = chunks(items, 14)
    for pi, part in enumerate(pages):
        d.text(d.W / 2, 35, "Ord sheet", 16, "B", "c")
        d.c.line(d.W / 2 - 45, d.y(38), d.W / 2 + 45, d.y(38))
        d.text(30, 55, "FAX NO", 8)
        d.text(230, 55, f"Ord dt :   {order_d:%Y-%m-%d}", 8)
        d.text(d.W - 120, 55, f"Page  :  {pi + 1}/{len(pages)}", 8)
        hc = [(55, "c"), (80, "c"), (70, "c"), (140, "c"), (60, "c"), (70, "c"), (50, "c"), (70, "c"), (60, "c"),
              (70, "c")]
        d.grid_row(25, 62, hc, ["Sply dt", sply_d.strftime("%Y-%m-%d"), "Ord slip no", po, "Ord fg", "Thông", "Batch",
                                "1 batch", "Sply tm", "00:00"], 16, 7.5)
        hc2 = [(20, "c"), (60, "c"), (130, "l"), (60, "c"), (80, "l"), (20, "c"), (60, "c"), (120, "l"), (60, "c"),
               (115, "c")]
        d.grid_row(25, 78, hc2, [["S", "P", "L", "Y"], ["Mst No.", "Comp nm"], ["1101912597", "CONG TY CO PHAN",
                                 "PHAN PHOI HANG TIEU"], ["Ven cd", "Nm"], ["008921", "NGUYEN", "QUOC"],
                                 ["R", "E", "C", "E"], ["Mst No.", "Comp nm"], ["0304741634-007", "CONG TY CP TRUNG",
                                 "TAM THUONG MAI"], ["Str/ nm", "Nm"], [store, city, "", "KIM MIN"]], 46, 7)
        cols = [(28, "c"), (70, "c"), (80, "c"), (190, "l"), (35, "c"), (35, "c"), (35, "c"), (45, "c"), (45, "r"),
                (60, "r"), (70, "r"), (40, "c")]
        top = 132
        d.grid_row(25, top, cols, ["Seq", "Prod cd", "Sale cd", "Prod nm", "Spec", "1-cat", "Uom", "Ord unit",
                                   "Ord qty", "Sply prc", "Sply amt", "Tax rt"], 16, 7.5)
        top += 16
        for k, it in enumerate(part):
            d.grid_row(25, top, cols, [str(pi * 14 + k + 1), it["prod"], it["code"], it["name"][:32], "", "55",
                                       str(it["pack"]), "BOX", str(it["qty"]), fmt_int(it["price"]),
                                       fmt_int(it["price"] * it["pack"] * it["qty"]), "8"], 18, 7.5)
            top += 18
        if pi == len(pages) - 1:
            tq = sum(i["qty"] for i in items)
            d.text(470, top + 20, f"{tq}      Sply amt", 7.5)
            d.text(660, top + 20, fmt_int(sum(i["price"] * i["pack"] * i["qty"] for i in items)), 7.5, align="r")
        d.end_page()
    d.save()


# ---------------------------------------------------------------- MM Mega Market (scanned: image only)

def mega(path, po, iln, order_d, deliv_d, items, comma_qty=False):
    d = Doc(path)
    d.text(d.W / 2, 40, "ĐƠN ĐẶT HÀNG", 8, "B", "c")
    d.text(470, 60, "MM  Mega Market", 13, "B", "c", Color(0.8, 0.1, 0.2))
    rows = [("Số thứ tự đơn đặt hàng", po), ("Ngày đặt hàng", f"{order_d:%d.%m.%Y}"),
            ("Giao hàng ngày", f"{deliv_d:%d.%m.%Y}"), ("Tài liệu tham khảo khuyến mãi", "")]
    for k, (a, b) in enumerate(rows):
        d.text(15, 70 + k * 11, a, 7, "B")
        d.text(145, 70 + k * 11, b, 7)
    d.text(240, 70, "Loại tài liệu O", 7, "B")
    blue = Color(0.0, 0.35, 0.45)
    for x, title, lines in [(15, "Người mua", [("GLN", "8935132500000"), ("Tên", "MM Mega Market VN"),
                                                ("Địa chỉ", "Khu B, Khu đô thị mới An Phú-An Khánh")]),
                            (210, "Người bán", [("ILN", "8934609000005"), ("Tên", "CTY CP PHAN PHOI HTD PAN"),
                                                ("Mã theo người mua:", "27594")]),
                            (405, "Địa điểm giao hàng", [("ILN", iln), ("Tên", "South Ambient Distribution Center"),
                                                         ("Địa chỉ", "Lô J2, Đường số 10, KCN Sóng Thần 1")])]:
        d.text(x + 90, 125, title, 7, "B", "c", blue)
        d.line(x, 128, x + 185, 128, 1)
        for k, (a, b) in enumerate(lines):
            d.text(x, 138 + k * 10, a, 6.5, "B")
            d.text(x + 55 if a != "Mã theo người mua:" else x + 95, 138 + k * 10, b, 6.5)
    d.text(105, 190, "Khách hàng", 7, "B", "c", blue)
    d.line(15, 193, 200, 193, 1)
    d.text(15, 203, "ILN", 6.5, "B")
    d.text(70, 203, "8935132500000", 6.5)
    cols = [(22, "c"), (66, "l"), (160, "l"), (50, "c"), (45, "r"), (45, "r"), (65, "r"), (110, "c")]
    light = Color(0.78, 0.88, 0.92)
    top = 225
    d.grid_row(10, top, cols, [["Dòng", "số"], "EAN sản phẩm", "Tên hàng hóa", ["Mã sản", "phẩm người", "mua"],
                               ["Đơn đặt", "hàng số", "lượng"], "Đơn giá", "Thành tiền",
                               "Tài liệu tham khảo khuyến mãi"], 32, 6.3, "B", fill=light)
    top += 32
    for k, it in enumerate(items):
        q = f"{it['qty']},00" if comma_qty else str(it["qty"])
        d.grid_row(10, top, cols, [str(k + 1), it["code"], it["name"][:32], it["buyer"], q,
                                   fmt_int(it["price"], "."), fmt_int(it["price"] * it["qty"], "."), ""], 11, 6.3)
        top += 11
    d.grid_row(430, top + 20, [(35, "c"), (50, "c"), (65, "c")], ["Số hàng", ["Tổng số", "lượng đặt"],
                                                                  "Thành tiền"], 24, 6.3, "B", fill=light)
    d.grid_row(430, top + 44, [(35, "r"), (50, "r"), (65, "r")],
               [str(len(items)), str(sum(i["qty"] for i in items)),
                fmt_int(sum(i["price"] * i["qty"] for i in items), ".")], 11, 6.3)
    d.end_page()
    d.save()


# ---------------------------------------------------------------- SATRA

def satra(path, po, order_d, deliv_d, items):
    d = Doc(path)
    usd = lambda x: f"{x.month}/{x.day}/{x.year}"  # noqa: E731 — month first, as printed by Satra
    d.text(110, 40, "Mẫu: MH-01", 10, "S")
    d.text(110, 58, "ĐƠN ĐẶT HÀNG", 13, "S")
    d.text(110, 74, f"Ngày:       {usd(order_d)}", 8, "S")
    d.text(380, 45, "Đơn đặt hàng số:", 10, "S")
    d.text(380, 68, po, 13, "SB")
    d.rect(25, 85, d.W - 50, 95)
    d.text(30, 100, "Đơn vị mua hàng:      Trung Tâm Điều Hành Satrafoods", 7.5, "S")
    d.text(30, 116, "Địa chỉ:     455 Võ Văn Tần, Phường 5, Quận 3, VNM", 7.5, "S")
    d.text(30, 132, "Mã số:      0300100037-025", 7.5, "S")
    d.text(30, 160, "Nơi giao hàng:         244 Lê Thị Hoa, KP.5, P.Bình Chiểu, TP. Thủ Đức,HCM,VNM", 7.5, "S")
    d.text(350, 100, "Điện thoại:    028.38329777", 7.5, "S")
    d.text(350, 116, "Người liên hệ:    Trương Minh Tiến", 7.5, "S")
    d.rect(25, 190, d.W - 50, 75)
    d.text(30, 206, "Đơn vị bán hàng:     CÔNG TY CỔ PHẦN PHÂN PHỐI HÀNG TIÊU DÙNG PAN", 7.5, "S")
    d.text(30, 222, "Địa chỉ:    Lô A1/1-9 Đường VL 3, KCN Vĩnh Lộc 2, Xã Long Hiệp, Huyện Bến Lức, Long An", 7.5, "S")
    d.text(30, 238, "Mã số:      1101912597", 7.5, "S")
    d.text(350, 206, "Mã số NCC: VD-00001986", 7.5, "S")
    cols = [(22, "r"), (42, "l"), (78, "l"), (190, "l"), (52, "l"), (45, "r"), (50, "r"), (58, "r"), (8, "l")]
    top = 280
    d.grid_row(25, top, cols, ["STT", "Mã hàng", "Barcode", "Tên hàng hóa", "Đơn vị", "Số lượng", "Đơn giá",
                               "Thành tiền", ""], 18, 7, "S", fill=GREY)
    top += 18
    for k, it in enumerate(items):
        d.grid_row(25, top, cols, [str(k + 1), it["mh"], it["satra_code"], it["satra_name"][:40], it["satra_unit"],
                                   fmt_dec(it["qty"], 3, ",", ","), fmt_dec(it["price"], 2, " ", ","),
                                   fmt_dec(it["price"] * it["qty"], 2, " ", ","), ""], 18, 7, "S")
        top += 18
    d.text(170, top + 14, "Tổng cộng", 7.5, "S")
    d.text(380, top + 14, fmt_dec(sum(i["qty"] for i in items), 3, ",", ","), 7.5, "S", "r")
    d.text(30, top + 40, "Hàng phục vụ cho:     MH", 7.5, "S")
    d.text(30, top + 54, f"Ngày đặt hàng:        {usd(order_d)}", 7.5, "S")
    d.text(30, top + 68, f"Ngày giao hàng:       {usd(deliv_d)}", 7.5, "S")
    d.text(30, d.H - 30, f"Ngày in: {usd(order_d)}", 7, "S")
    d.end_page()
    d.save()


# ---------------------------------------------------------------- VIN (WinCommerce)

def vin(path, po, point, point_name, order_d, deliv_d, items):
    d = Doc(path)
    pages = chunks(items, 16)
    for pi, part in enumerate(pages):
        d.text(d.W - 25, 25, f"Trang {pi + 1} / {len(pages)}", 7, align="r")
        d.text(d.W / 2, 60, "Đơn đặt hàng / Purchase Order", 11, "B", "c")
        if pi == 0:
            d.text(25, 95, "Thông tin xuất hóa đơn (Billing Information)", 7.5, "B")
            d.text(25, 107, "Chi nhánh - Công ty Cổ phần Dịch vụ Thương mại Tổng hợp Wincommerce", 7)
            d.text(25, 119, "MST: 0104918404-056", 7)
            d.text(25, 135, "Địa chỉ giao hàng (Delivery Address)", 7.5, "B")
            d.text(25, 147, f"{point}-{point_name}", 7)
            d.text(25, 159, f"{point} - {point_name}", 7)
            d.text(25, 183, "Nhà cung cấp (Supplier): 0002062500", 7)
            d.text(25, 195, "CTY CP HÀNG TIÊU DÙNG PAN", 7)
            d.text(310, 95, "Thông tin đơn hàng (Information)", 7.5, "B")
            for k, (a, b) in enumerate([("Số đơn hàng (PO No.)", po), ("Ngày đặt hàng (PO date)", f"{order_d:%d.%m.%Y}"),
                                        ("Nhóm đặt hàng (Purchaser)", "001 - ĐH qua DC(VM/VM+)"),
                                        ("Người đặt hàng (Purchaser)", "System User"),
                                        ("Ngày giao (Delivery Date)", f"{deliv_d:%d.%m.%Y}")]):
                d.text(310, 107 + k * 12, a, 7)
                d.text(450, 107 + k * 12, b, 7)
            top = 215
        else:
            top = 80
        cols = [(30, "c"), (170, "l"), (80, "c"), (55, "r"), (35, "c"), (60, "r"), (90, "r")]
        d.grid_row(25, top, cols, [["Stt", "(No.)"], ["Tên hàng", "(Item Description)"], ["Mã vạch", "(Barcode)"],
                                   ["Số lượng", "(Quantity)"], ["ĐVT", "(Unit)"], ["Đơn giá", "(Unit Price)"],
                                   ["Thành tiền (VND)", "(Amount)"]], 22, 6.8, "B", border=False)
        top += 26
        for k, it in enumerate(part):
            n = (pi * 16 + k + 1) * 10
            d.grid_row(25, top, cols, [str(n), [it["vin_item"], it["name"][:34]], it["code"], fmt_int(it["qty"]),
                                       "G1", fmt_int(it["price"]), fmt_int(it["price"] * it["qty"])], 24, 6.8,
                       border=False)
            top += 24
        if pi == len(pages) - 1:
            tv = sum(i["price"] * i["qty"] for i in items)
            d.text(25, top + 16, "Tổng giá trị trước thuế (Total excl VAT amount)", 7)
            d.text(d.W - 30, top + 16, fmt_int(tv), 7, align="r")
        d.end_page()
    d.save()


# ---------------------------------------------------------------- LAN CHI (printed spreadsheet)

def lanchi(path, order_d, items):
    d = Doc(path, landscape(A4))
    d.text(25, 30, "Tên siêu thị: THE CITY", 9, "S")
    d.text(25, 44, "Địa chỉ giao hàng: Tổng kho The City, Số 25 - Ngã tư Cổng Ô – Phường Lê Lợi – Thị xã Sơn Tây "
                   "– Hà Nội", 9, "S")
    d.text(25, 58, f"Ngày đặt          {order_d:%d/%m/%Y}", 9, "S")
    d.text(190, 74, "ĐƠN ĐẶT HÀNG - SIÊU THỊ THE CITY", 10, "SB")
    stores = ["Yên M", "Yên Đị", "Thiệu H", "Triệu S", "Hoàng H"]
    cols = [(52, "l"), (75, "l"), (220, "l"), (45, "r"), (40, "c"), (65, "r"), (45, "r")] + [(45, "r")] * 5
    d.grid_row(20, 92, [(c[0], c[1]) for c in cols], [""] * 7 + [str(i + 1) for i in range(5)], 12, 7, "S",
               border=False)
    d.grid_row(20, 104, cols, ["Mã hàng", "Mã vạch", "Tên hàng", "Giá mua", "CTKM", "Thành Tiền", "Tổng đặt"] + stores,
               14, 7, "SB")
    top = 118
    for it in items:
        split = it["split"]
        d.grid_row(20, top, cols, [it["lc"], it["code"], it["name"][:44], fmt_int(it["price"]),
                                   it.get("ctkm", ""), fmt_int(it["price"] * it["qty"]), str(it["qty"])]
                   + [str(x) if x else "-" for x in split], 14, 7, "S")
        top += 14
    d.end_page()
    d.save()


# ---------------------------------------------------------------- scans

def rasterize(path, dpi=150, angle=0.0, noise=0, quality=80):
    """Replace a PDF by images of its pages (no text layer), optionally skewed / blurred / noisy."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    imgs = []
    for page in doc:
        im = page.render(scale=dpi / 72).to_pil().convert("L")
        if angle:
            im = im.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
        if noise:
            im = im.filter(ImageFilter.GaussianBlur(0.6))
            px = im.load()
            r = random.Random(7)
            for _ in range(im.width * im.height // 60):
                x, y = r.randrange(im.width), r.randrange(im.height)
                px[x, y] = max(0, min(255, px[x, y] + r.randint(-noise, noise)))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=quality)
        imgs.append(Image.open(io.BytesIO(buf.getvalue())))
    doc.close()
    imgs[0].save(path, "PDF", save_all=True, append_images=imgs[1:], resolution=dpi)


# ---------------------------------------------------------------- build

def items_for(n, qty_range=(1, 6)):
    out = []
    for code, name in pick(n):
        pack = rng.choice([10, 12, 20, 24, 30, 40, 60, 80, 100])
        out.append({"code": code, "name": name, "qty": rng.randint(*qty_range), "pack": pack,
                    "price": rng.randint(30, 700) * 100 + rng.choice([0, 0, 50]),
                    "unit": rng.choice(["Hộp", "Túi", "Gói"]), "lv": rng.randint(0, 3)})
    return out


def gt(stem, customer, fields, note=""):
    d = {"file": stem, "customer": customer, "checked": True, "synthetic": True, "fields": fields}
    if note:
        d["note"] = note
    return d


def build():
    pdf_dir, gt_dir = OUT / "pdf", OUT / "gt"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    gts = []
    D = date

    # AEON — image-only (real Aeon PDFs have an unusable text layer)
    for k, (store, sname, od, n, scan) in enumerate([
            ("1005", "HA DONG", D(2024, 3, 4), 4, dict()),
            ("1004", "AEON TAN PHU", D(2024, 7, 11), 11, dict(angle=0.6)),
            ("8003", "AEON HCM RDC (XD)", D(2024, 12, 30), 17, dict(angle=-0.8, noise=60, quality=55))], 1):
        stem = f"AEON_synth_{k}"
        po = f"{store}{rng.randint(4000000000, 4999999999)}"
        dd = od + timedelta(days=rng.choice([1, 2, 3]))
        its = items_for(n)
        for it in its:
            it["ncc"] = f"0139{rng.randint(1000, 9999)}"
            it["price"] = rng.randint(100000, 900000) / 1000
        aeon(pdf_dir / f"{stem}.pdf", po, store, sname, od, dd, its)
        rasterize(pdf_dir / f"{stem}.pdf", **scan)
        gts.append(gt(stem, "AEON", {"Số PO": po, "Ngày đề nghị giao hàng": dd.isoformat(), "Mã KH": store,
                                     "Chi tiết hàng": [{"Mã SP": i["code"], "SL đặt": i["qty"]} for i in its]}))

    # BHX — kho codes from the DMS sheet; style 2 has the internal code on the top line (2021 layout)
    for k, (kho, kname, od, n, old) in enumerate([
            ("6450", "BHX_DLA_BMT - Kho DC Buôn Ma Thuột", D(2024, 2, 5), 3, False),
            ("5851", "BHX_BDU_TAN - Kho DC Thuận An", D(2024, 5, 9), 9, True),
            ("6655", "BHX_HCM_NBE - Kho DC Nhà Bè", D(2024, 11, 28), 16, False)], 1):
        stem = f"BHX_synth_{k}"
        po = f"0{kho}PO{od:%y%m}{rng.randint(100000, 999999)}"
        its = items_for(n, (2, 40))
        for it in its:
            it["qty"] = it["qty"] * it["pack"] // rng.choice([1, 2]) or it["pack"]
            it["internal"] = f"10530{rng.randint(10000000, 99999999)}"
            it["carton"] = "8934609400" + f"{rng.randint(100, 999)}"
        d2 = od + timedelta(days=6)
        bhx(pdf_dir / f"{stem}.pdf", po, kho, kname, od, od, d2, its, old_style=old)
        gts.append(gt(stem, "BHX", {"Số PO": po, "Ngày đề nghị giao hàng": od.isoformat(), "Mã KH": kho,
                                    "Chi tiết hàng": [{"Mã SP": i["internal"] if old else i["code"],
                                                       "SL đặt": i["qty"], "Đơn vị": i["unit"]} for i in its]}))

    # BigC — PURCHASE NOTE pages (extract) + DETAIL DELIVERY BY STORE pages (must be ignored)
    for k, (contract, dname, od, n, n_stores) in enumerate([
            ("M1500806", "FM LOGISTIC VSIP 2 (806)", D(2024, 1, 29), 6, 1),
            ("M1500212", "KHO BIG C BINH DUONG (212)", D(2024, 4, 3), 22, 3),
            ("M1500121", "CTY TNHH EB BINH DUONG (121)", D(2024, 12, 28), 45, 4)], 1):
        stem = f"BigC_synth_{k}"
        po = f"23{rng.randint(10 ** 10, 10 ** 11 - 1)}"
        its = items_for(n, (1, 20))
        stores = []
        for s in ["SIEU THI DA NANG", "SIEU THI HUE", "SIEU THI THANG LONG", "SIEU THI DONG NAI"][:n_stores]:
            stores.append((s, [(it, rng.randint(1, max(1, it["qty"]))) for it in rng.sample(its, min(len(its), 12))]))
        bigc(pdf_dir / f"{stem}.pdf", po, contract, dname, od, od + timedelta(days=6), its, stores)
        gts.append(gt(stem, "BigC", {"Số PO": po, "Ngày đề nghị giao hàng": (od + timedelta(days=5)).isoformat(),
                                     "Mã KH": contract,
                                     "Chi tiết hàng": [{"Mã SP": i["code"], "SL đặt": i["qty"]} for i in its]},
                      "Order Date + 5 ngày (quy tắc Excel)"))

    # COOP
    for k, (loc, ed, n) in enumerate([("801", D(2024, 3, 7), 5), ("812", D(2024, 8, 2), 14),
                                      ("830", D(2024, 12, 30), 31)], 1):
        stem = f"COOP_synth_{k}"
        po = f"8{rng.randint(1000000, 9999999)}-00"
        exp = ed + timedelta(days=rng.choice([4, 6, 7]))
        its = items_for(n, (10, 200))
        for it in its:
            it["sku"] = f"3{rng.randint(000000, 999999):06d}-{rng.randint(0, 9)}"
            it["short"] = it["name"].title().replace(" ", "")[:28] if rng.random() < 0.3 else it["name"].title()
            it["vpn"] = rng.choice(["A8", "", f"18{rng.randint(10 ** 12, 10 ** 13 - 1)}"])
            it["price"] = rng.randint(4000, 60000)
        coop(pdf_dir / f"{stem}.pdf", po, loc, ed, exp, its)
        gts.append(gt(stem, "COOP", {"Số PO": po, "Ngày đề nghị giao hàng": exp.isoformat(), "Mã KH": loc,
                                     "Chi tiết hàng": [{"Mã SP": i["sku"], "SL đặt": i["qty"]} for i in its]}))

    # E-Mart — Mã KH is the whole "Delivery to" text
    for k, (deliv, od, n) in enumerate([
            ("EMART GO VAP 366 PHAN VAN TRI,P5,Q.GO VAP,TPHCM", D(2024, 2, 1), 2),
            ("EMART SALA 10 MAI CHI THO,P.AN LOI DONG,TP.THU DUC", D(2024, 6, 12), 8),
            ("EMART PHAN HUY ICH 2 PHAN HUY ICH,P.15,Q.TAN BINH", D(2024, 10, 9), 19)], 1):
        stem = f"EMART_synth_{k}"
        po = f"45{rng.randint(10 ** 7, 10 ** 8 - 1)}"
        dd = od + timedelta(days=2)
        its = items_for(n, (1, 9))
        for it in its:
            it["article"] = f"10{rng.randint(10000, 99999)}"
            it["price"] = rng.randint(100, 900) * 1000 + rng.randint(0, 999)
        emart(pdf_dir / f"{stem}.pdf", po, deliv, od, dd, its)
        gts.append(gt(stem, "E-Mart", {"Số PO": po, "Ngày đề nghị giao hàng": dd.isoformat(), "Mã KH": deliv,
                                       "Chi tiết hàng": [{"Mã SP": i["code"], "SL đặt": i["qty"]} for i in its]}))

    # Lotte — store codes with leading zeros
    for k, (store, city, od, n) in enumerate([("01013", "Nha Trang", D(2024, 1, 8), 3),
                                              ("01004", "Da Nang", D(2024, 6, 3), 12),
                                              ("01015", "NT Gold", D(2024, 9, 10), 24)], 1):
        stem = f"Lotte_synth_{k}"
        po = f"{od:%y%m%d}-{store}-{rng.randint(10, 999):05d}"
        sd = od + timedelta(days=3)
        its = items_for(n, (1, 5))
        for it in its:
            it["prod"] = f"1-{rng.randint(1000, 199999):06d}-000"
            it["price"] = rng.randint(4000, 40000)
        lotte(pdf_dir / f"{stem}.pdf", po, store, city, od, sd, its)
        gts.append(gt(stem, "Lotte", {"Số PO": po, "Ngày đề nghị giao hàng": sd.isoformat(), "Mã KH": store,
                                      "Chi tiết hàng": [{"Mã SP": i["code"], "SL đặt": i["qty"]} for i in its]}))

    # MM — image-only scans; file 2 prints quantities as "60,00"
    for k, (iln, od, n, comma, scan) in enumerate([
            ("8935132500710", D(2024, 4, 2), 3, False, dict()),
            ("8935132500710", D(2024, 8, 9), 10, True, dict(angle=0.5)),
            ("8935132500727", D(2024, 12, 27), 22, False, dict(noise=50, quality=50))], 1):
        stem = f"MM_synth_{k}"
        po = f"90071.{rng.randint(10 ** 7, 10 ** 8 - 1)}"
        dd = od + timedelta(days=3)
        its = items_for(n, (2, 30))
        for it in its:
            it["qty"] *= 10
            it["buyer"] = str(rng.randint(100000, 499999))
            it["price"] = rng.randint(3000, 45000)
        mega(pdf_dir / f"{stem}.pdf", po, iln, od, dd, its, comma_qty=comma)
        rasterize(pdf_dir / f"{stem}.pdf", **scan)
        gts.append(gt(stem, "MM", {"Số PO": po, "Ngày đề nghị giao hàng": dd.isoformat(), "Mã KH": iln,
                                   "Chi tiết hàng": [{"Mã SP": i["code"], "SL đặt": i["qty"]} for i in its]}))

    # SATRA — month-first dates (one ambiguous: 4/3), decimal comma quantities, carton-prefixed barcodes
    for k, (od, n) in enumerate([(D(2024, 4, 3), 2), (D(2024, 7, 25), 6), (D(2024, 12, 29), 12)], 1):
        stem = f"SATRA_synth_{k}"
        po = f"P-00{rng.randint(4000000, 4999999)}"
        its = items_for(n, (1, 4))
        for it in its:
            it["mh"] = str(rng.randint(110000, 249999))
            if rng.random() < 0.7:
                it["satra_unit"] = f"THÙNG{it['pack']}"
                it["satra_code"] = f"{it['pack']}{it['code']}"
                it["satra_name"] = f"{it['name']} X {it['pack']}"
                it["price"] = rng.randint(150000, 700000) + 0.0
            else:
                it["satra_unit"] = rng.choice(["HŨ", "GÓI", "HỘP"])
                it["satra_code"] = it["code"]
                it["satra_name"] = it["name"]
                it["qty"] = it["qty"] * 12
                it["price"] = rng.randint(8000, 40000) + 0.0
        satra(pdf_dir / f"{stem}.pdf", po, od, od + timedelta(days=1), its)
        gts.append(gt(stem, "SATRA", {"Số PO": po, "Ngày đề nghị giao hàng": (od + timedelta(days=5)).isoformat(),
                                      "Mã KH": "VD-00001986",
                                      "Chi tiết hàng": [{"Mã SP": i["satra_code"], "SL đặt": i["qty"]} for i in its]},
                      "'Ngày' (tháng/ngày/năm) + 5 ngày (quy tắc Excel)"))

    # VIN
    for k, (pt, pname, od, n) in enumerate([("1324", "WCM_AMBIENT_DA NANG_HANGTHUONG", D(2024, 3, 12), 1),
                                            ("1333", "WMT_AMBIENT_BINH DUONG2_HANGTH", D(2024, 6, 6), 7),
                                            ("1331", "WMT_AMBIENT_HUNGYEN_HANGTHUONG&KM", D(2024, 11, 5), 20)], 1):
        stem = f"VIN_synth_{k}"
        po = f"41{rng.randint(10 ** 7, 10 ** 8 - 1)}"
        dd = od + timedelta(days=rng.choice([5, 8]))
        its = items_for(n, (12, 2400))
        for it in its:
            it["vin_item"] = f"100{rng.randint(10000, 99999)}"
            it["price"] = rng.randint(4000, 45000)
        vin(pdf_dir / f"{stem}.pdf", po, pt, pname, od, dd, its)
        gts.append(gt(stem, "VIN", {"Số PO": po, "Ngày đề nghị giao hàng": dd.isoformat(), "Mã KH": pt,
                                    "Chi tiết hàng": [{"Mã SP": i["code"], "SL đặt": i["qty"]} for i in its]}))

    # LAN CHI — no real sample: layout of the Excel screenshot; no PO number; Ngày đặt + 5
    for k, (od, n) in enumerate([(D(2024, 2, 7), 4), (D(2024, 9, 13), 10), (D(2024, 12, 30), 18)], 1):
        stem = f"LANCHI_synth_{k}"
        its = items_for(n)
        for it in its:
            split = [rng.choice([0, 0, 6, 12, 30, 60]) for _ in range(5)]
            if not any(split):
                split[0] = 6
            it["split"], it["qty"] = split, sum(split)
            it["lc"] = f"00{rng.randint(100000, 299999)}"
            it["price"] = rng.randint(4000, 140000)
            it["ctkm"] = rng.choice(["", "", "10%", "15%"])
        lanchi(pdf_dir / f"{stem}.pdf", od, its)
        gts.append(gt(stem, "LANCHI", {"Ngày đề nghị giao hàng": (od + timedelta(days=5)).isoformat(),
                                       "Mã KH": "THE CITY",
                                       "Chi tiết hàng": [{"Mã SP": i["code"], "SL đặt": i["qty"]} for i in its]},
                      "Ngày đặt + 5 ngày (quy tắc Excel); bố cục theo ảnh trong Excel, chưa có PDF thật"))

    for g in gts:
        (gt_dir / f"{g['file']}.json").write_text(json.dumps(g, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(gts)} synthetic POs -> {pdf_dir.relative_to(ROOT)} , ground truth -> {gt_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    build()
