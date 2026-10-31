# Verification log: raw_TULD_MULTI.jsonl -> verified_TULD_MULTI.jsonl

Input 105 questions; output 92.

- TULD: unchanged 37, fixed 13, dropped 0
- MULTI: unchanged 24, fixed 6, dropped 0
- NONE: unchanged 12, fixed 0, dropped 13

## Changed / dropped

| id | action | reason |
|---|---|---|
| TULD-006 | fixed | key_fact was a 7-word phrase; shortened to "hiệu lực của hợp đồng" |
| TULD-020 | fixed | key_fact "30 tháng 06 năm 2024" not required in a correct answer (may give only the end date); now "3 năm", "90 ngày", "2027" |
| TULD-027 | fixed | question asks only about the first offence; removed key_fact "lần thứ 2" |
| TULD-029 | fixed | question not anchored although QCL Điều 12 has the same rule (gold would be incomplete); anchored to thỏa ước |
| TULD-030 | fixed | question not anchored; QCL Điều 13 adds an extra 1% BHTN payment for ≥14-day leave, so a full answer would need QCL too; anchored to thỏa ước |
| TULD-032 | fixed | question not anchored; QCL Điều 17 has the identical rule (gold would be incomplete); anchored to thỏa ước |
| TULD-040 | fixed | key_fact "tái ký HĐLĐ" risky (HĐLĐ may be spelled out); reduced to "tái ký" |
| TULD-042 | fixed | question not anchored; QCL Điều 21 (nghỉ mát hàng năm từ quỹ khen thưởng) would also answer it; anchored to thỏa ước |
| TULD-043 | fixed | question not anchored; QCL Điều 21 also covers rewards for improvement ideas; anchored to thỏa ước |
| TULD-044 | fixed | answer wrongly said Thỏa ước has no specific policy; TULD Điều 11 gives a company maternity allowance; added Điều 11 |
| TULD-045 | fixed | ambiguous ("where was it signed" vs company address; two parties with two addresses); clarified to the employer address |
| TULD-046 | fixed | key_fact "quy định của pháp luật" too generic; replaced by "bảo vệ dữ liệu cá nhân" |
| TULD-047 | fixed | removed risky key_fact "phụ lục số 3" (can be written Phụ lục 03) |
| MULTI-001 | fixed | question compares QCL and TULD only; removed superfluous NQLD Điều 12 from answer and gold |
| MULTI-006 | fixed | removed key_fact "½ tháng lương công việc" (symbol ½ likely written 1/2) |
| MULTI-009 | fixed | question asks Nội quy + Quy chế only; removed superfluous TULD Điều 7 from answer and gold |
| MULTI-017 | fixed | the 30 phút/ngày (≤03 ngày/tháng) break is not a pregnancy/child-care entitlement; removed from answer and key_facts |
| MULTI-020 | fixed | "có hiệu lực kể từ ngày ký" is stated in TULD Điều 19 (added to gold); key_fact "Sở Lao động Thương binh và Xã hội" often abbreviated, replaced |
| MULTI-021 | fixed | closing provisions now in node NQLD/Chương IX (no longer NQLD/Điều 32); updated gold and citation |
| NONE-002 | dropped | weak near-miss (generic WFH question); trimmed to keep 12 strongest unanswerables |
| NONE-003 | dropped | weak near-miss (gym benefit, no related text); trimmed |
| NONE-004 | dropped | weak near-miss (ESOP); trimmed |
| NONE-006 | dropped | out-of-domain (company revenue), not an HR near-miss; trimmed |
| NONE-008 | dropped | weak near-miss (private health insurance); trimmed |
| NONE-015 | dropped | weak near-miss (housing); trimmed |
| NONE-019 | dropped | weak near-miss (housing loan); trimmed |
| NONE-020 | dropped | weak near-miss (24/24 accident insurance); trimmed |
| NONE-021 | dropped | partially answerable: NQLD Điều 13d forbids any personal work during working hours, a correct system would answer rather than refuse |
| NONE-022 | dropped | weak near-miss (referral bonus, no related text); trimmed |
| NONE-023 | dropped | answerable: TULD Điều 17 says trip budget depends on business results (duplicate of TULD-042) |
| NONE-024 | dropped | weak near-miss (uniform count); trimmed |
| NONE-025 | dropped | weak near-miss (2025 Tết calendar); trimmed |
