from bibica_rag.po.spec import detect_customer, dump_specs, find_spec_file, infer_type, load_specs

LONG = [
    ["DANH SÁCH THÔNG TIN TRÍCH XUẤT PO"],  # merged title row above the real header
    [],
    ["STT", "Khách hàng", "Thông tin cần trích xuất", "Mô tả", "Kiểu dữ liệu", "Nhóm", "Từ khóa nhận diện"],
    [1, "Co.opmart", "Số PO", "Mã đơn hàng góc trên phải", "", "", "SAIGON CO.OP; coop"],
    [2, None, "Ngày đặt hàng", "", "ngày", "", None],
    [3, None, "Mã hàng", "", "", "Chi tiết hàng hóa", None],
    [4, None, "Số lượng", "", "", "Chi tiết hàng hóa", None],
    [5, "Bách Hóa Xanh", "Số PO", "", "", "", "BHX"],
    [6, None, "Tổng tiền", "Tổng cộng sau VAT", "", "", None],
]


def test_long_layout(xlsx_factory):
    specs = load_specs(xlsx_factory({"Sheet1": LONG}))
    assert list(specs) == ["Co.opmart", "Bách Hóa Xanh"]
    coop = specs["Co.opmart"]
    assert coop.field_names == ["Số PO", "Ngày đặt hàng", "Chi tiết hàng hóa"]
    assert coop.fields[0].description == "Mã đơn hàng góc trên phải"
    assert coop.fields[1].type == "date"
    table = coop.fields[2]
    assert table.type == "table" and [c.name for c in table.columns] == ["Mã hàng", "Số lượng"]
    assert table.columns[1].type == "number"
    assert coop.aliases == ["SAIGON CO.OP", "coop"]
    bhx = specs["Bách Hóa Xanh"]
    assert bhx.field_names == ["Số PO", "Tổng tiền"] and bhx.fields[1].type == "number"


def test_matrix_layout(xlsx_factory):
    rows = [["Thông tin", "Mô tả", "Co.opmart", "Bách Hóa Xanh", "Lotte"],
            ["Số PO", "", "x", "x", "x"],
            ["Ngày giao hàng", "", "x", "", "✓"],
            ["Địa chỉ giao hàng", "", "", "1", None]]
    specs = load_specs(xlsx_factory({"Fields": rows}))
    assert specs["Co.opmart"].field_names == ["Số PO", "Ngày giao hàng"]
    assert specs["Bách Hóa Xanh"].field_names == ["Số PO", "Địa chỉ giao hàng"]
    assert specs["Lotte"].field_names == ["Số PO", "Ngày giao hàng"]


def test_wide_layout(xlsx_factory):
    rows = [["Khách hàng", "Số PO", "Ngày PO", "Tổng tiền"],
            ["Co.opmart", "x", "x", ""],
            ["Lotte", "PO-123 (góc phải)", "", "x"]]
    specs = load_specs(xlsx_factory({"KH": rows}))
    assert specs["Co.opmart"].field_names == ["Số PO", "Ngày PO"]
    assert specs["Lotte"].field_names == ["Số PO", "Tổng tiền"]
    assert specs["Lotte"].fields[0].example == "PO-123 (góc phải)"


def test_sheet_per_customer(xlsx_factory):
    path = xlsx_factory({
        "Coop": [["Khách hàng: Co.opmart"], ["STT", "Trường", "Ghi chú"], [1, "Số PO", "x"], [2, "Kho nhận", ""]],
        "Lotte": [["STT", "Trường"], [1, "Số đơn hàng"]],
    })
    specs = load_specs(path)
    assert specs["Co.opmart"].field_names == ["Số PO", "Kho nhận"]
    assert specs["Lotte"].field_names == ["Số đơn hàng"]


def test_yaml_override_next_to_excel(xlsx_factory, tmp_path):
    path = xlsx_factory({"Sheet1": LONG})
    path.with_suffix(".yaml").write_text(
        "customers:\n  - name: Lotte\n    aliases: [LOTTE MART]\n    fields:\n      - Số PO\n"
        "      - {name: Hàng, columns: [{name: Mã SP}, {name: SL, type: number}]}\n", encoding="utf-8")
    specs = load_specs(path)
    assert set(specs) == {"Co.opmart", "Bách Hóa Xanh", "Lotte"}
    assert specs["Lotte"].fields[1].type == "table"
    assert specs["Lotte"].fields[1].columns[1].type == "number"
    # round trip through YAML keeps everything
    out = tmp_path / "rt.yaml"
    out.write_text(dump_specs(specs), encoding="utf-8")
    assert {k: v.to_dict() for k, v in load_specs(out).items()} == {k: v.to_dict() for k, v in specs.items()}
    assert find_spec_file(tmp_path) == path


def test_infer_type():
    assert infer_type("Số PO") == "text"
    assert infer_type("Mã số thuế") == "text"
    assert infer_type("Số lượng") == "number"
    assert infer_type("Đơn giá") == "number"
    assert infer_type("Ngày giao hàng") == "date"
    assert infer_type("Địa chỉ giao hàng") == "text"


def test_detect_customer(xlsx_factory):
    specs = load_specs(xlsx_factory({"Sheet1": LONG}))
    assert detect_customer(specs, "PO Extraction/Coopmart/PO_123.pdf")[0] == "Co.opmart"
    assert detect_customer(specs, "PO Extraction/BHX_0425.pdf")[0] == "Bách Hóa Xanh"
    # nothing in the path -> text layer
    name, why = detect_customer(specs, "scan001.pdf", "LIEN HIEP HTX TM TP.HCM - SAIGON CO.OP\nPURCHASE ORDER")
    assert name == "Co.opmart" and "nội dung" in why
    # short alias must be a whole word: 'bhx' inside 'abhxy' is not a match
    assert detect_customer(specs, "abhxy.pdf")[0] is None
    # file name beats text
    assert detect_customer(specs, "BHX/po.pdf", "Saigon Co.op")[0] == "Bách Hóa Xanh"
