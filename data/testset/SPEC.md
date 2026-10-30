# Test-set spec (Bibica RAG benchmark)

Source documents (Markdown, already parsed, authoritative):
- data/parsed/QCL.md  — Quy chế tiền lương, tiền thưởng Bibica 2024 (code QCL)
- data/parsed/NQLD.md — Nội quy lao động 2024 (code NQLD)
- data/parsed/TULD.md — Thỏa ước lao động tập thể 2024 (code TULD)
Each section header carries its node id in an HTML comment, e.g. `<!-- QCL/Điều 12 -->`.
Valid node ids: `<DOC>/Mở đầu`, `<DOC>/Điều N`, `QCL/Phụ lục N` (N without leading zero).
Images: data/images/*.{png,jpeg} (placeholders `[HÌNH ẢNH: name]` in the md). QCL_image1 = Bibica logo;
the others are digital-signature stamps (signer name, signing date/time) — open them with the Read tool.

Output: one JSON object per line (JSONL, UTF-8, ensure_ascii=False), fields:
- id: "<PREFIX>-NNN"
- question: Vietnamese question as a real employee/HR staff would ask
- type: one of fact | numeric_table | multi_hop | cross_doc | conditional | list | unanswerable | paraphrase | no_diacritics | definition
- doc: "QCL" | "NQLD" | "TULD" | "MULTI" (needs >1 doc) | "NONE" (unanswerable)
- answerable: true/false
- reference_answer: complete, correct Vietnamese answer, grounded ONLY in the documents, citing the article(s) e.g. "(QCL Điều 12)". For unanswerable: state that the documents do not contain this information.
- gold_nodes: list of node ids containing the evidence needed (ALL nodes needed for a full answer; [] for unanswerable)
- key_facts: 0-4 short atomic strings (1-5 words) that MUST appear in any correct answer, copied in the document's wording — numbers with unit/percent ("150%", "12 ngày", "06 tháng"), specific named terms ("khiển trách", "sa thải"). Prefer numbers/percentages/names. Do not include facts that could legitimately be paraphrased in many ways. [] for unanswerable.

Type meanings:
- fact: single-hop lookup of one statement
- numeric_table: numbers, percentages, values from tables/appendices (KPI weights, scoring scales, coefficients)
- multi_hop: needs combining 2+ places within one document (or a small calculation)
- cross_doc: compares/combines 2+ documents, incl. where QCL and TULD (or NQLD) say the same or different things
- conditional: scenario "nếu nhân viên ... thì ..." applying a rule to a case
- list: enumerate all items of a list (all forms of discipline, all allowances...)
- unanswerable: plausible HR question whose answer is NOT in the documents (must be truly absent — double check); the system must refuse
- paraphrase: colloquial/indirect wording, no keywords copied from the text
- no_diacritics: Vietnamese written without diacritics and/or with typos/abbreviations (e.g. "nghi phep nam duoc may ngay")
- definition: meaning of a term defined in the documents

Rules: every answerable question must be fully answerable from the documents; avoid ambiguity; vary difficulty;
cover ALL articles and appendices across the set; no duplicate questions; questions must not mention node ids.
