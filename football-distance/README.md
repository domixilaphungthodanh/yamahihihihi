# Dự đoán quãng đường di chuyển của cầu thủ sau trận

Dự án hồi quy có giám sát, một dòng là một cầu thủ trong một trận World Cup nam 2022. Mục tiêu là tổng quãng đường di chuyển theo km. Bốn nhóm vị trí: thủ môn, hậu vệ, tiền vệ, tiền đạo.

## Nguồn và quyền sử dụng

- [FIFA Training Centre: 64 báo cáo trận](https://www.fifatrainingcentre.com/en/fwc2022/post-match-summaries/post-match-summary-reports.php): vị trí, quãng đường và thống kê trận. PDF có lỗi mã hóa chữ số khi trích text, nên chương trình OCR các trang Physical Data và ghép theo mã trận, đội, số áo.
- [Fjelstul World Cup Database qua DataHub](https://datahub.io/football/worldcup): thời điểm thay người và cờ hiệp phụ; [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- [Điều khoản FIFA](https://www.fifatrainingcentre.com/en/terms-of-service.php): repository chỉ có mã, tài liệu và đường dẫn nguồn. PDF, bảng dữ liệu trích xuất, mô hình và báo cáo chứa dữ liệu theo từng cầu thủ được giữ cục bộ.

Số phút trên sân là ước tính theo thời điểm thay người và trận dài 90 hoặc 120 phút danh nghĩa, có sai lệch do bù giờ. Dữ liệu các giải khác chưa được kiểm chứng.

## Chuẩn bị Windows

Từ PowerShell trong `E:\machine-learing`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r football-distance\requirements.txt
```

Cần [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) và `pdftoppm` của [Poppler](https://poppler.freedesktop.org/) để chuẩn bị dữ liệu. Trên máy đã triển khai, Tesseract ở `C:\Program Files\Tesseract-OCR\tesseract.exe`, Poppler có trong runtime Codex. Trên máy khác, đặt `pdftoppm` và `tesseract` vào PATH hoặc khai báo `PDFTOPPM_EXE` và `TESSERACT_EXE` là đường dẫn tuyệt đối.

```powershell
.\.venv\Scripts\python.exe football-distance\src\project.py prepare
.\.venv\Scripts\python.exe football-distance\src\project.py train
.\.venv\Scripts\python.exe -m notebook football-distance\notebooks\football_distance.ipynb
.\.venv\Scripts\python.exe football-distance\src\project.py predict --input football-distance\examples\player.json
.\.venv\Scripts\python.exe -m unittest discover -s football-distance\tests -v
```

`prepare` tải và đọc 64 PDF, có thể mất nhiều phút. Dữ liệu được cache cục bộ. Notebook đọc dữ liệu đã chuẩn bị và gọi cùng hàm huấn luyện với CLI. Nếu PowerShell chặn `.ps1`, các lệnh trên vẫn chạy trực tiếp qua Python.

## Phương pháp

`distance_km` lấy từ tổng mét di chuyển chia 1000. Đầu vào gồm `position`, `minutes_played`, `started`, `extra_time`, `team_goals`, `opponent_goals`, `team_passes`, `team_shots`. Không dùng các cột quãng đường hoặc vùng tốc độ làm đặc điểm. Baseline dùng tốc độ di chuyển trung vị theo vị trí, nhân thời gian chơi. Ridge và Random Forest được so sánh bằng MAE qua GroupKFold 5 fold trên train. 20% trận được giữ làm test, seed 42; không chia cầu thủ cùng trận sang hai tập.

Các file `reports/extraction_issues.csv` và `reports/extracted_matches.csv` ghi tỷ lệ thành công. Sau huấn luyện, `reports/results.md`, `reports/position_metrics.csv` và `reports/evaluation.png` cho biết kết quả thực tế. MAE càng thấp càng tốt, đơn vị km. Các nhóm vị trí có số mẫu khác nhau; luôn đọc MAE cùng cột `n`.

### Kết quả chạy trên máy này

64 báo cáo được đọc, 1.911 dòng cầu thủ hợp lệ. 51 trận (1.517 dòng) dùng để huấn luyện; 13 trận (394 dòng) giữ làm test. Ridge được chọn bằng cross-validation. Test MAE **0,564 km**, RMSE **0,811 km**, R² **0,945**; baseline test MAE **0,682 km**. MAE theo vị trí: thủ môn 0,882 km (27 mẫu), hậu vệ 0,540 km (123), tiền vệ 0,573 km (147), tiền đạo 0,492 km (97). Các giá trị sẽ thay đổi nếu nguồn trực tuyến hoặc phiên bản thư viện thay đổi.

## Giới hạn

Đây là ước tính sau trận, không phải dự báo trước trận. Vị trí được rút gọn thành bốn nhóm và không phản ánh vai trò chiến thuật chi tiết. Chỉ có các trận World Cup 2022; không tuyên bố độ chính xác cho giải khác. OCR và ghép bảng có thể lỗi; các dòng không xác thực bị loại và được ghi trong báo cáo. Khi dùng mô hình, chỉ tải file `joblib` bạn tin tưởng.
