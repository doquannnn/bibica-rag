# Judge rubric — Bibica RAG benchmark

You judge answers produced by RAG systems to employee questions about 3 internal HR documents of Bibica
(QCL = Quy chế tiền lương tiền thưởng 2024, NQLD = Nội quy lao động 2024, TULD = Thỏa ước lao động tập thể 2024).
Full documents: data/parsed/QCL.md, NQLD.md, TULD.md (node ids in `<!-- ... -->`). Each batch item gives the question,
the verified reference answer, key facts, the gold evidence text, and one or more anonymous answers (each with a `key`).
Images in the md files appear only as `[HÌNH ẢNH: file]` placeholders; their content is part of the documents and is in
data/parsed/image_captions.json: NQLD (Chương IX) is digitally signed by Nguyễn Quốc Hoàng on 22/03/2024 22:45:07
(lần ký 1); QCL (after Điều 23) is signed by Nguyễn Quốc Hoàng on 26/04/2024 21:32:24 (lần ký 4); QCL_image1 is the
Bibica logo. Answers using these facts are supported (the typed date line "Ngày …… tháng ……" is blank, the stamp is not).
You do not know which system wrote which answer — judge each answer on its own merits.

For EVERY answer output one JSON line:
`{"key": "...", "qid": "...", "correctness": 1 | 0.5 | 0, "faithful": true|false, "citation_ok": true|false|null, "refusal_ok": true|false|null, "note": "<= 20 words"}`

- **correctness** (vs reference answer and gold evidence):
  - 1 = all information the question asks for is present and correct (numbers, conditions, every list item, both sides of a comparison). Extra correct detail is fine. Wording may differ.
  - 0.5 = partially correct: main point right but a requested element missing or slightly wrong, or a list incomplete.
  - 0 = wrong, contradicts the documents, answers a different question, or refuses although the answer exists.
  - Unanswerable questions (answerable=false): 1 if the answer clearly says the documents do not contain this
    (mentioning genuinely related content is fine), 0 if it invents an answer / presents outside knowledge as company policy.
  - Documents that disagree: several questions do not name a document although QCL / TULD / NQLD regulate the
    point differently (e.g. pay day: QCL ngày 15 vs TULD ngày 20 + ngày 10). If the question does not name a
    document, an answer that correctly states another document's provision (with that document named) is not
    wrong: give 0.5 if it gives only the other document's rule, 1 if it also gives the reference's rule or explains
    the difference. If the question names a document, judge against that document only.
- **faithful**: false if the answer states ANY concrete claim (number, rule, condition, attribution to a document)
  that is not supported by the documents — check the gold evidence first, and grep the full md files when the
  answer cites other parts. Generic connecting sentences are fine. A wrong number is unfaithful.
- **citation_ok**: for answers that cite sources ([QCL Điều 12] etc.): true if the cited articles actually contain the
  cited content (minor numbering slips on one of several citations → still true); false if citations are mostly wrong
  or fabricated; null if the answer cites nothing.
- **refusal_ok**: only for answerable=false items: true if it correctly declines; otherwise null.

Be strict and consistent; do not reward length. Vietnamese answers without diacritics are fine if correct.
