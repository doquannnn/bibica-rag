# Verification log — NQLD

Source: data/testset/raw_NQLD.jsonl (version md5 ce75fffe…, 100 questions; the file was regenerated at 21:28 during review and this log covers that version) -> data/testset/verified_NQLD.jsonl.

Totals: 45 unchanged, 55 fixed, 0 dropped.

Checks done for all 100: answerability vs data/parsed/NQLD.md, reference correctness (calculations in 037, 038, 039, 043, 053 re-done), gold node minimality (incl. new `NQLD/Chương IX` node — 025/026 already use it; no question still points to Điều 32 for closing-provision/signature evidence), key-fact necessity/robustness.

## Changed

- NQLD-003: key_facts -> ['Thương binh và Xã hội'] (agency name hyphen/abbrev variants; dropped 'tất cả người lao động' (paraphrasable: toàn thể...))
- NQLD-004: key_facts -> ['Thương binh và Xã hội', 'xác nhận đăng ký'] (shortened agency name (hyphen/abbrev variants))
- NQLD-005: key_facts -> ['thứ bảy', 'chủ nhật'] (dropped '04 ngày nghỉ/tháng' (not asked); shortened day ranges)
- NQLD-010: key_facts -> ['ốm đau'] (dropped 'BHXH' (may be spelled out 'bảo hiểm xã hội'))
- NQLD-014: key_facts -> ['Tổng Giám đốc', 'quản lý trực tiếp'] (shortened)
- NQLD-015: key_facts -> ['văn minh', 'lịch sự', 'khiếm nhã'] (split 'văn minh lịch sự' (comma variant))
- NQLD-016: key_facts -> ['sạch sẽ', 'gọn gàng', 'lịch sự'] (split comma list into atomic facts)
- NQLD-018: key_facts -> ['vệ sinh nơi làm việc', 'thiết bị điện'] ('tắt, khóa' has comma + khóa/khoá variant)
- NQLD-020: key_facts -> ['quản lý', 'Giám đốc'] (split 'người quản lý hoặc Giám đốc')
- NQLD-023: key_facts -> ['giao kết hợp đồng', 'đại diện'] ('Người đại diện pháp luật' often written 'đại diện theo pháp luật'; shortened)
- NQLD-038: key_facts -> ['12 ngày'] (dropped '16 ngày' (input value, not the answer); arithmetic 16/12x9=12 verified)
- NQLD-039: key_facts -> ['50%', '6.000.000 đồng', 'trả đủ'] ('trả đủ tiền lương' may be 'trả đủ lương'; 50%x12.000.000=6.000.000 verified)
- NQLD-040: key_facts -> ['khiển trách', 'nâng lương', '6 tháng'] (missing core sanction 'nâng lương' added; dropped '3 tháng' (threshold, not required); doc uses both 'thời gian/thời hạn nâng lương')
- NQLD-045: gold: removed superfluous Điều 17 (Điều 28.4.a names trộm cắp directly); answer citation adjusted
- NQLD-046: gold: [Điều 8, 26, 28] -> [Điều 28] (28.3.d alone answers; Điều 8 irrelevant, Điều 26 only restates); type multi_hop -> conditional; answer reworded; key_facts -> ['cách chức'] (dropped 'Giả mạo chứng nhận' (echoes question, paraphrasable))
- NQLD-047: key_facts -> ['ca đêm'] (dropped 'tháng thứ 7' (from question) and 'có thai' (may be 'mang thai'))
- NQLD-048: key_facts -> ['nâng lương', '6 tháng'] (dropped '02 lần' (given in question); thời gian/thời hạn nâng lương variant)
- NQLD-050: key_facts -> ['nâng lương', '6 tháng'] (thời gian/thời hạn nâng lương variant)
- NQLD-051: key_facts -> ['sa thải'] (dropped '05 ngày' (given in question))
- NQLD-052: key_facts -> ['sa thải'] (dropped '20 ngày' (given in question) and '365 ngày' (may be 'một năm'))
- NQLD-053: key_facts -> ['03 tháng', '27.000.000 đồng', 'khấu trừ'] ('03 tháng tiền lương' may be '03 tháng lương'; hàng/hằng tháng variant; 3x9.000.000=27.000.000 verified)
- NQLD-056: key_facts -> ['không có lý do chính đáng', 'nâng lương'] (thời gian/thời hạn nâng lương variant)
- NQLD-057: key_facts -> ['cách chức'] (dropped 'nặng nhất' (reasoning wording, paraphrasable))
- NQLD-058: key_facts -> ['phạt tiền', 'cắt lương'] (split comma pair)
- NQLD-059: key_facts -> ['30 tháng 3'] (dropped 'năm kế tiếp' (may be 'năm sau'))
- NQLD-061: key_facts -> ['07 ngày', 'xác nhận', 'đủ lương'] (question also asks about pay -> added 'đủ lương'; dropped '04 tháng' (precondition already met in question))
- NQLD-062: key_facts -> ['sa thải'] (dropped '06 tháng' (reasoning step, 6/06 variant))
- NQLD-063: key_facts -> ['không được từ chối'] (dropped 'hỏa hoạn' (given in question))
- NQLD-064: key_facts -> ['khiển trách', 'nâng lương', 'cách chức', 'sa thải'] (doc uses both 'thời gian'/'thời hạn' nâng lương)
- NQLD-065: key_facts -> ['lập biên bản', 'thu thập chứng cứ', 'họp xử lý kỷ luật', 'ban hành quyết định'] (shortened/lowercased)
- NQLD-066: key_facts -> ['tạm giam', 'nghỉ thai sản', '12 tháng', 'bệnh tâm thần'] ('tạm giữ, tạm giam' comma risk; shortened)
- NQLD-067: key_facts -> ['nhân phẩm', 'phạt tiền', 'không được quy định'] (split comma pairs)
- NQLD-069: key_facts -> ['sơ cứu', 'bảo hộ lao động', 'định kỳ', 'bảng chỉ dẫn'] ('khám sức khỏe' khỏe/khoẻ variant; shortened)
- NQLD-070: key_facts -> ['danh tiếng', 'phạm vi trách nhiệm', 'giả mạo', 'lừa đảo'] (lowercased/shortened)
- NQLD-071: key_facts -> ['hút thuốc', 'đánh bạc', 'ma túy', 'dữ liệu cá nhân'] (lowercased)
- NQLD-072: key_facts -> ['danh nghĩa công ty', 'cản trở giao dịch', 'giả mạo', 'lừa đảo'] (lowercased/shortened)
- NQLD-073: key_facts -> ['10 tháng lương', 'làm mất', 'tiêu hao vật tư'] ('tối thiểu vùng' suffix may be omitted; lowercased)
- NQLD-076: key_facts -> ['không được ngủ'] ('ngủ trong thời gian làm việc' may be 'trong giờ làm việc')
- NQLD-077: key_facts -> ['tình tiết phức tạp'] (dropped 'tổ chức đại diện tập thể lao động' (Điều 27 legitimately calls it 'công đoàn cơ sở'))
- NQLD-080: key_facts -> ['rời bỏ'] ('từ chối hoặc rời bỏ' shortened)
- NQLD-082: key_facts -> ['bí mật'] (dropped 'không có quyền hạn' (paraphrasable))
- NQLD-083: key_facts -> ['đồng ý', 'quốc phòng', 'thiên tai'] (shortened 'đồng ý của người lao động' and comma pair 'quốc phòng, an ninh')
- NQLD-084: key_facts -> ['Công ty'] ('Công ty quyết định' may be 'công ty sẽ quyết định')
- NQLD-085: key_facts -> ['1 ngày'] (dropped '05 năm' (given in question); '1 ngày' also matches '01 ngày')
- NQLD-086: key_facts -> ['thay đổi'] ('thay đổi ngày nghỉ hàng tuần' long, hàng/hằng variant)
- NQLD-087: key_facts -> ['ốm'] ('ốm đau' may be 'nghỉ ốm')
- NQLD-088: key_facts -> ['3 năm'] (dropped 'không bị coi là tái phạm' (paraphrasable); '3 năm' also matches '03 năm')
- NQLD-089: key_facts -> [] ('thanh toán bằng tiền' freely paraphrasable ('được trả tiền'))
- NQLD-091: key_facts -> ['vẫn tiến hành'] (shortened)
- NQLD-092: key_facts -> ['khiển trách'] (dropped '5.000.000 đồng' (threshold, may be '5 triệu'))
- NQLD-094: key_facts -> ['bình thường'] ('ngoài thời giờ làm việc bình thường' may be 'ngoài giờ làm việc...')
- NQLD-095: key_facts -> ['thiên tai', 'cơ sở y tế'] (comma pair; hỏa/hoả variant)
- NQLD-096: key_facts -> ['10 tháng lương', '30 tháng lương'] ('tối thiểu vùng' suffix may be omitted)
- NQLD-097: key_facts -> ['03 lần', '03 tháng liên tục', '03 kỳ đánh giá'] (shortened)
- NQLD-100: key_facts -> ['ngày tháng năm sinh', 'số điện thoại', 'giới tính', 'bằng cấp'] ('Căn cước công dân' may be abbreviated CCCD)

## Dropped

- none (no question was wrong or ambiguous beyond repair)
