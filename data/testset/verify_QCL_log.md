# verify_QCL log

Input: data/testset/raw_QCL.jsonl (105 rows, md5 00c9698d…, version written 21:28:05). Output: data/testset/verified_QCL.jsonl (105 rows).

Unchanged: 69 | Fixed: 36 | Dropped: 0

Coverage note: no question targets QCL/Mở đầu or QCL/Phụ lục 1 (Phụ lục 1 only says the pay scale is attached separately).

## Changed / dropped ids
- QCL-004: FIXED — removed key_fact "phụ lục 01" (not required for a correct definition)
- QCL-008: FIXED — "cùng một ngạch lương" -> "ngạch lương" (paraphrase-safe)
- QCL-011: FIXED — "70 ÷ 85%"/"15 ÷ 30" use ÷ sign (answer may write 70-85%); split into plain numbers
- QCL-017: FIXED — multi-item key_facts ("Cử nhân, kỹ sư, trưởng ca") order/case-sensitive; split to atomic terms
- QCL-018: FIXED — scenario "bậc 2 + 31 năm -> bậc 4" ambiguous ("nâng lên đúng bậc" vs "mỗi kỳ chỉ được nâng 01 bậc" could mean bậc 3); changed to bậc 3 start so answer bậc 4 is unambiguous
- QCL-022: FIXED — "nhu cầu sản xuất, kinh doanh" -> "nhu cầu sản xuất" (comma variant)
- QCL-024: FIXED — "01 – 05" (en dash) format-sensitive; split into "01","05"
- QCL-025: FIXED — "100% tiền lương theo HĐLĐ" too long/paraphrasable -> "100%"
- QCL-028: FIXED — "Quỹ lương - thưởng bổ sung" hyphen/spacing variant -> "thưởng bổ sung"
- QCL-034: FIXED — "14 ngày đầu tiên" -> "14 ngày"
- QCL-035: FIXED — "trả đủ tiền lương" -> "trả đủ" (answer may say "trả đủ lương")
- QCL-037: FIXED — "ngày làm việc kế tiếp" -> "kế tiếp"
- QCL-038: FIXED — "Công ty cổ phần Bibica" paraphrasable (Công ty Bibica) -> "Bibica"
- QCL-043: FIXED — removed "6 tháng" (given in question); added "BHXH" (base differs from BHTN case)
- QCL-044: FIXED — "2 triệu đồng" -> "2 triệu"
- QCL-046: FIXED — "100% ngày công" -> "100%"
- QCL-047: FIXED — "100% ngày công" -> "100%"
- QCL-048: FIXED — "80 ÷ 90%"/"10 ÷ 20" (÷ sign) split into plain numbers
- QCL-049: FIXED — removed "10 ÷ 40" (÷ sign format-sensitive); kept "40"
- QCL-050: FIXED — capitalised list items ("Hệ số...", "Ngày công...") case-sensitive; trimmed to case-neutral substrings
- QCL-051: FIXED — "Kỹ sư, Cử nhân" order-sensitive -> "Kỹ sư"
- QCL-052: FIXED — "chỉ tăng lương cơ bản" paraphrasable -> "lương cơ bản"
- QCL-054: FIXED — "700.000đ" -> "700.000" (đ/đồng variants)
- QCL-055: FIXED — "800.000đ" -> "800.000"
- QCL-056: FIXED — calc verified (900.000/25x23=828.000); "900.000đ" -> "900.000"
- QCL-059: FIXED — calc verified (10.000x60=600.000); "10.000 đồng" -> "10.000"
- QCL-066: FIXED — "tỉ lệ thời gian làm việc" (tỉ/tỷ spelling) -> "thời gian làm việc"
- QCL-069: FIXED — removed "03 ngày" (threshold, not required; 3/03 variant)
- QCL-073: FIXED — "26 tháng 04 năm 2024" vs 26/04/2024 both valid -> "26"; "kể từ ngày ký" -> "ngày ký"
- QCL-090: FIXED — "tháng 17, 18" format-sensitive -> "17","18"
- QCL-092: FIXED — "01 tháng lương công việc" -> "01 tháng"; "công thực tế / công chuẩn" spacing-sensitive -> "công chuẩn"
- QCL-093: FIXED — removed "½ tháng lương công việc" (½ vs 1/2/nửa); added formula term "công chuẩn"
- QCL-094: FIXED — original scenario had profit 95% (<100%): unclear whether Công ty "đạt chỉ tiêu năm về lợi nhuận" (PL3 4a) -> ambiguous; changed to 106%/102% (=108 điểm), recomputed
- QCL-098: FIXED — removed "Trung bình" (level name not asked)
- QCL-101: FIXED — "Mức độ hoàn thành công việc" -> "hoàn thành công việc"
- QCL-105: FIXED — removed "Đặc biệt quan trọng" (question asks only the score)
