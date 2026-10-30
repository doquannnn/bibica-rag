import json

from bibica_rag.po.extract import (build_prompt, build_schema, extract_po, parse_json_response, postprocess)
from bibica_rag.po.spec import CustomerSpec, Field

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


def test_extract_end_to_end_with_fake_vlm(pdf_factory, fake_client):
    pdf = pdf_factory("scan_0001.pdf", [COOP_PAGE, ["page 2"]])
    reply = {"Số PO": "4500123", "Ngày đặt hàng": "2025-03-05", "Tổng tiền": 15500000,
             "Chi tiết hàng hóa": [{"Mã hàng": "8934567", "Số lượng": 120}, {"Mã hàng": "8934999", "Số lượng": 1000}]}
    client = fake_client(reply)
    r = extract_po(pdf, SPECS, client=client, model="m")
    assert r.error == ""
    assert r.customer == "Co.opmart" and "nội dung" in r.customer_reason  # found via the PDF text layer
    assert r.fields == reply and r.missing == [] and r.pages_used == 2
    call = client.calls[0]
    assert call["temperature"] == 0
    assert call["response_format"]["json_schema"]["schema"] == build_schema(SPECS["Co.opmart"])
    content = call["messages"][1]["content"]
    assert sum(c["type"] == "image_url" for c in content) == 2
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "PO No: 4500123" in content[-1]["text"]  # text layer passed as a hint


def test_extract_forced_customer_and_schema_fallback(pdf_factory, fake_client):
    pdf = pdf_factory("x.pdf", [["SO DON HANG: L-77", "KHO: Binh Duong"]])
    client = fake_client(RuntimeError("response_format json_schema not supported"),
                         '{"Số đơn hàng": "L-77", "Kho nhận": "Binh Duong", "Số PO": "nope"}')
    r = extract_po(pdf, SPECS, customer="Lotte", client=client, model="m", use_text_layer=False)
    assert r.fields == {"Số đơn hàng": "L-77", "Kho nhận": "Binh Duong"} and r.dropped_keys == ["Số PO"]
    assert "response_format" not in client.calls[1]
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
