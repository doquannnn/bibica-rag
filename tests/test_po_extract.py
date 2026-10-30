import json

from bibica_rag.po.dms import load_dms_mapping
from bibica_rag.po.extract import (build_prompt, build_schema, extract_po, parse_json_response, postprocess,
                                   text_is_usable)
from bibica_rag.po.spec import DEFAULT_SPEC, CustomerSpec, Field, load_specs

SPECS = {
    "Co.opmart": CustomerSpec("Co.opmart", [
        Field("Số PO"), Field("Ngày đặt hàng", "date"), Field("Tổng tiền", "number"),
        Field("Chi tiết hàng hóa", "table", columns=[Field("Mã hàng"), Field("Số lượng", "number")]),
    ], aliases=["SAIGON CO.OP"]),
    "Lotte": CustomerSpec("Lotte", [Field("Số đơn hàng"), Field("Kho nhận")]),
}

COOP_PAGE = ["SAIGON CO.OP - PURCHASE ORDER", "PO No: 4500123", "Date: 05/03/2025",
             "Item 8934567 Qty 120", "Item 8934999 Qty 1,000", "Total: 15.500.000 VND"]


def test_prompt_and_schema_only_contain_that_customers_fields():
    prompt = build_prompt(SPECS["Co.opmart"], "PO No: 4500123")
    assert '"Số PO"' in prompt and '"Mã hàng"' in prompt and "4500123" in prompt
    assert "Số đơn hàng" not in prompt and "Kho nhận" not in prompt
    schema = build_schema(SPECS["Co.opmart"])
    assert list(schema["properties"]) == ["Số PO", "Ngày đặt hàng", "Tổng tiền", "Chi tiết hàng hóa"]
    assert schema["additionalProperties"] is False and schema["required"] == list(schema["properties"])
    items = schema["properties"]["Chi tiết hàng hóa"]["items"]
    assert list(items["properties"]) == ["Mã hàng", "Số lượng"] and items["additionalProperties"] is False


def test_parse_json_response_tolerates_noise():
    assert parse_json_response('<think>hmm</think>```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('Kết quả: {"a": "x"} xong') == {"a": "x"}


def test_postprocess_keeps_exactly_the_spec_fields():
    data = {"so po": "4500123", "Tổng tiền": "15.500.000 VND", "Nhà cung cấp": "Bibica", "Ghi chú": "abc",
            "Chi tiết hàng hóa": [{"Mã hàng": "8934567", "Số lượng": "120", "Đơn giá": 5},
                                  {"Mã hàng": None, "Số lượng": None}]}
    fields, missing, dropped = postprocess(SPECS["Co.opmart"], data)
    assert list(fields) == SPECS["Co.opmart"].field_names
    assert fields["Số PO"] == "4500123"  # key matched ignoring case/accents
    assert fields["Tổng tiền"] == 15500000
    assert fields["Ngày đặt hàng"] is None and missing == ["Ngày đặt hàng"]
    assert fields["Chi tiết hàng hóa"] == [{"Mã hàng": "8934567", "Số lượng": 120}]  # extra column + empty row gone
    assert dropped == ["Nhà cung cấp", "Ghi chú"]


def test_extract_end_to_end_multi_page(pdf_factory, fake_client):
    pdf = pdf_factory("scan_0001.pdf", [COOP_PAGE, ["SAIGON CO.OP - PURCHASE ORDER (continued)", "Item 8935000000001 Qty 7"]])
    page1 = {"Số PO": "4500123", "Ngày đặt hàng": "2025-03-05", "Tổng tiền": 15500000,
             "Chi tiết hàng hóa": [{"Mã hàng": "8934567", "Số lượng": 120}, {"Mã hàng": "8934999", "Số lượng": 1000}]}
    page2 = {"Chi tiết hàng hóa": [{"Mã hàng": "8935000000001", "Số lượng": "7"}]}
    client = fake_client(page1, page2)
    r = extract_po(pdf, SPECS, client=client, model="m")
    assert r.error == ""
    assert r.customer == "Co.opmart" and "nội dung" in r.customer_reason  # found via the PDF text layer
    assert r.text_layer == "used" and r.pages_used == [1, 2] and r.missing == []
    # header fields from page 1, line items of both pages concatenated in order
    assert r.fields == {**page1, "Chi tiết hàng hóa": page1["Chi tiết hàng hóa"] + [
        {"Mã hàng": "8935000000001", "Số lượng": 7}]}
    first, second = client.calls
    assert first["temperature"] == 0
    assert first["response_format"]["json_schema"]["schema"] == build_schema(SPECS["Co.opmart"])
    # later pages are only asked for the tables
    assert list(second["response_format"]["json_schema"]["schema"]["properties"]) == ["Chi tiết hàng hóa"]
    content = first["messages"][1]["content"]
    assert sum(c["type"] == "image_url" for c in content) == 1  # one page image per call
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "PO No: 4500123" in content[-1]["text"] and "Item 8935000000001" not in content[-1]["text"]


def test_skip_pages_and_date_offset(pdf_factory, fake_client):
    spec = {"BigC": CustomerSpec("BigC", [Field("Ngày đề nghị giao hàng", "date", offset_days=5),
                                          Field("Hàng", "table", columns=[Field("Mã SP"), Field("SL", "number")])],
                                 skip_pages_containing=["DETAIL DELIVERY BY STORE"])}
    pdf = pdf_factory("BigC_x.pdf", [["PURCHASE NOTE", "Order Date 28/02/24 15:01"],
                                     ["DETAIL DELIVERY BY STORE", "8934609203093 1"],
                                     ["PURCHASE NOTE", "8934609105939 1"]])
    client = fake_client({"Ngày đề nghị giao hàng": "2024-02-28", "Hàng": [{"Mã SP": "A", "SL": 1}]},
                         {"Hàng": [{"Mã SP": "B", "SL": 2}]})
    r = extract_po(pdf, spec, client=client, model="m")
    assert r.pages_used == [1, 3] and len(client.calls) == 2
    assert r.fields["Ngày đề nghị giao hàng"] == "2024-03-04"  # +5 days, across the leap day
    assert [x["Mã SP"] for x in r.fields["Hàng"]] == ["A", "B"]


def test_garbled_text_layer_is_not_used():
    assert text_is_usable("P/O Number: 81131529-00 P/O Location: 801 Expected Receipt Date - 6/11/23")
    garbage = "[GE67 I>\\E ' SBJ[6ED H] $[^ &2 9c6 <=67 STORE ORDER - _ ` 0 ( S a ` S b d e 6 $UB 7=@ 6 Bf 6"
    assert not text_is_usable(garbage)
    assert not text_is_usable("")


def test_dms_lookup_after_extraction(pdf_factory, fake_client, xlsx_factory):
    path = xlsx_factory({"BHX": [[None, None, None, None, "Đơn vị tính túi hộp DMS"],
                                 ["TÊN SIÊU THỊ", "MÃ TRÊN PO PDF", "MÃ DMS", "TÊN SIÊU THỊ DMS"],
                                 ["BHX DC LONG BÌNH", 4089, 9352, "9352 - BHX_DON_BHO - Kho DC Long Bình"]],
                         "Lotte": [["TÊN SIÊU THỊ", "MÃ TRÊN PO PDF", "MÃ DMS", "TÊN SIÊU THỊ DMS"],
                                   ["Lotte Dong Nai", "01003", "6814", "LOTTE ĐỒNG NAI"]]})
    mapping = load_dms_mapping(path)
    specs = {"BHX": CustomerSpec("BHX", [Field("Mã KH")], dms_sheet="BHX", code_field="Mã KH"),
             "Lotte": CustomerSpec("Lotte", [Field("Mã KH")], dms_sheet="Lotte", code_field="Mã KH")}
    pdf = pdf_factory("BHX_1.pdf", [["BHX order"]])
    r = extract_po(pdf, specs, client=fake_client({"Mã KH": "4089"}), model="m", dms_mapping=mapping)
    assert r.dms["Mã DMS"] == "9352" and r.dms["Đơn vị tính DMS"] == "túi hộp DMS"
    assert list(r.fields) == ["Mã KH"]  # DMS data is kept apart from the extracted fields
    r = extract_po(pdf, specs, customer="Lotte", client=fake_client({"Mã KH": "1003"}), model="m", dms_mapping=mapping)
    assert r.dms["Mã DMS"] == "6814"  # leading zeros ignored
    r = extract_po(pdf, specs, client=fake_client({"Mã KH": "7211"}), model="m", dms_mapping=mapping)
    assert r.dms["Mã DMS"] is None and "7211" in r.dms["ghi_chu"]


def test_bibica_config():
    specs = load_specs(DEFAULT_SPEC)
    assert {"AEON", "BHX", "BigC", "COOP", "Lotte", "E-Mart", "MM", "LANCHI", "SATRA", "VIN"} == set(specs)
    for name, spec in specs.items():
        core = ["Ngày đề nghị giao hàng", "Mã KH", "Chi tiết hàng"] if name == "LANCHI" else \
            ["Số PO", "Ngày đề nghị giao hàng", "Mã KH", "Chi tiết hàng"]
        assert spec.field_names == core, name
        items = spec.fields[-1]
        assert [c.name for c in items.columns][:2] == ["Mã SP", "SL đặt"] and items.columns[1].type == "number"
        assert spec.dms_sheet and spec.code_field == "Mã KH"
        assert all(f.description for f in spec.fields[:-1]), name
    assert [c.name for c in specs["BHX"].fields[-1].columns] == ["Mã SP", "SL đặt", "Đơn vị"]
    assert {n for n, s in specs.items() if s.fields[-3].offset_days == 5} == \
        {"BigC", "LANCHI", "SATRA"}


def test_extract_forced_customer_and_schema_fallback(pdf_factory, fake_client):
    pdf = pdf_factory("x.pdf", [["SO DON HANG: L-77", "KHO: Binh Duong"]])
    client = fake_client(RuntimeError("response_format json_schema not supported"),
                         '{"Số đơn hàng": "L-77", "Kho nhận": "Binh Duong", "Số PO": "nope"}')
    r = extract_po(pdf, SPECS, customer="Lotte", client=client, model="m", use_text_layer=False)
    assert r.fields == {"Số đơn hàng": "L-77", "Kho nhận": "Binh Duong"} and r.dropped_keys == ["Số PO"]
    assert "response_format" not in client.calls[1] and r.text_layer == "disabled"
    assert "SO DON HANG" not in client.calls[1]["messages"][1]["content"][-1]["text"]


def test_extract_falls_back_to_vlm_customer_classification(pdf_factory, fake_client):
    pdf = pdf_factory("scan.pdf", [["(no recognisable header)"]])
    client = fake_client({"customer": "lotte"}, {"Số đơn hàng": "1", "Kho nhận": None})
    r = extract_po(pdf, SPECS, client=client, model="m")
    assert r.customer == "Lotte" and r.customer_reason.startswith("VLM")
    assert client.calls[0]["response_format"]["json_schema"]["schema"]["properties"]["customer"]["enum"] == \
        ["Co.opmart", "Lotte", None]
    assert r.missing == ["Kho nhận"]


def test_extract_reports_errors_instead_of_raising(pdf_factory, fake_client):
    pdf = pdf_factory("scan.pdf", [["?"]])
    r = extract_po(pdf, SPECS, client=fake_client({"customer": None}), model="m")
    assert r.customer is None and "LookupError" in r.error
    r = extract_po(pdf, SPECS, customer="Lotte", client=fake_client("không phải json"), model="m")
    assert "ValueError" in r.error
    assert json.loads(json.dumps(r.to_dict()))["file"].endswith("scan.pdf")
