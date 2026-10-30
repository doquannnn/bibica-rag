from bibica_rag.rag.chat import Retriever, answer_stream, build_messages

NODES = [
    {"node_id": "NQLD/Điều 10", "title": "Điều 10. Nghỉ hằng năm", "doc_title": "Nội quy lao động",
     "text": "Người lao động làm việc đủ 12 tháng được nghỉ phép năm 12 ngày làm việc, hưởng nguyên lương."},
    {"node_id": "QCL/Điều 12", "title": "Điều 12. Thưởng tháng 13", "doc_title": "Quy chế tiền lương",
     "text": "Thưởng tháng lương thứ 13 được chi trả trước Tết Nguyên đán."},
    {"node_id": "TULD/Điều 5", "title": "Điều 5. Ăn ca", "doc_title": "Thỏa ước",
     "text": "Tiền ăn giữa ca theo quy định của công ty."},
]


def test_retrieval_with_and_without_diacritics():
    r = Retriever(NODES)
    assert r.search("Nghỉ phép năm được mấy ngày?", k=1)[0].node["node_id"] == "NQLD/Điều 10"
    assert r.search("nghi phep nam duoc may ngay", k=1)[0].node["node_id"] == "NQLD/Điều 10"
    assert r.search("thuong thang 13", k=1)[0].node["node_id"] == "QCL/Điều 12"
    assert r.search("xyzzy") == []


def test_answer_stream_grounds_and_streams(fake_client):
    r = Retriever(NODES)
    client = fake_client("Được nghỉ 12 ngày (NQLD Điều 10).")
    history = [{"role": "user", "content": "xin chào"}, {"role": "assistant", "content": "Chào bạn"}]
    outs = list(answer_stream("nghỉ phép năm", r, history, k=2, client=client, model="m"))
    assert outs[-1][0] == "Được nghỉ 12 ngày (NQLD Điều 10)." and len(outs) > 1
    msgs = client.calls[0]["messages"]
    assert msgs[0]["role"] == "system" and msgs[1]["content"] == "xin chào"
    assert "[NQLD/Điều 10]" in msgs[-1]["content"] and client.calls[0]["stream"] is True


def test_build_messages_without_hits():
    assert "không tìm thấy" in build_messages("abc", [])[-1]["content"]
