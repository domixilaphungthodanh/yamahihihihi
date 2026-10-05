# Dự đoán tiêu hao nhiên liệu của xe

Dự án học Supervised Learning / Regression bằng Python, pandas, NumPy, Matplotlib và scikit-learn. Notebook giải thích bằng tiếng Việt, dùng cùng module Python với dòng lệnh.

## Chạy trên Windows PowerShell

Môi trường `.venv` đã được chuẩn bị trên máy này. Không cần kích hoạt môi trường:

**Trạng thái:** Đã huấn luyện thành công, chạy notebook từ đầu đến cuối và vượt qua 5 kiểm thử. Mô hình và báo cáo đã có trong `models/` và `reports/`.

```powershell
cd E:\machine-learing
.\.venv\Scripts\python.exe -m notebook notebooks\fuel_consumption.ipynb
```

Chọn kernel **Fuel ML (.venv)**, sau đó dùng **Restart Kernel and Run All Cells**. Notebook huấn luyện lại mô hình nên cần chờ vài phút.

```powershell
# Huấn luyện và tạo báo cáo
.\.venv\Scripts\python.exe src\fuel.py train
# Dự đoán xe mẫu
.\.venv\Scripts\python.exe src\fuel.py predict --input examples\vehicle.json
# Kiểm thử
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Cài lại trên máy khác

Cài Python 3.12, rồi chạy trong thư mục dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name fuel-ml --display-name "Fuel ML (.venv)"
```

## Nhập thông tin xe

Sửa bản sao của `examples/vehicle.json`. Các trường: `cylinders` (số xi-lanh), `displacement_l` (lít), `horsepower` (hp), `weight_kg` (kg), `acceleration_s` (giây tăng tốc 0–60 mph), `model_year` (năm đầy đủ), `origin` (`usa`, `europe`, `japan`). Không nhập giá trị theo đơn vị inch khối hoặc pound. Công suất có thể để `null` hoặc bỏ trường, pipeline sẽ điền bằng median học từ train.

Đầu ra `fuel_l_per_100km` là số lít dự đoán để đi 100 km trong điều kiện dữ liệu nguồn. `warnings` cho biết đầu vào ngoài khoảng train; đây không phải khoảng tin cậy thống kê.

## Tổ chức và phương pháp

- `data/raw`: dữ liệu UCI gốc, archive và mô tả nguồn; `data/processed`: dữ liệu chuyển đơn vị.
- `notebooks`: notebook học từng bước; `src`: module và CLI.
- `models`: pipeline và metadata; `reports`: kết quả, dự đoán test, importance và biểu đồ.
- `tests`: kiểm tra dữ liệu, chuyển đơn vị và dự đoán; `examples`: xe mẫu.

80% train, 20% test, seed 42. Chọn mô hình bằng MAE cross-validation 5 fold trên train. Test chỉ dùng sau khi chọn mô hình. So sánh baseline trung bình, Ridge và Random Forest; preprocessing nằm trong pipeline để tránh rò rỉ dữ liệu. MAE và RMSE tính bằng L/100 km; MAE nhỏ hơn tốt hơn. R² được báo cáo bổ sung, có thể âm nếu mô hình kém.

Đọc `reports/results.md` để xem kết quả thực tế. Mô hình lưu là mô hình fit trên train, đúng với đánh giá test; không tự fit lại trên toàn bộ dữ liệu.

## Giới hạn

Dữ liệu lịch sử xe 1970–1982, mức tiêu hao chu trình đô thị. Không suy ra tiêu hao chuyến đi thực tế, xe điện hay đảm bảo cho xe hiện đại. Dữ liệu không có hộp số hoặc tắc đường. Permutation importance trên train mang tính khám phá, không chứng minh nguyên nhân. Chia ngẫu nhiên không kiểm chứng dự báo năm tương lai. Chỉ tải file joblib do bạn tin tưởng.

## Nguồn

UCI Auto MPG: https://archive.ics.uci.edu/dataset/9/auto+mpg (CC BY 4.0 theo trang UCI). Giữ nguyên dữ liệu gốc; dữ liệu processed đổi đơn vị và bổ sung mục tiêu L/100 km. MPG dùng US gallon: L/100 km = 235.214583 / MPG.

## Bài tập tiếp theo

Thử thay đổi đặc điểm hoặc cấu hình bằng cross-validation trên train, so sánh kết quả và giải thích lỗi. Khi đã xem test nhiều lần để điều chỉnh, tập test không còn là đánh giá độc lập; hãy tạo một tập đánh giá mới cho nghiên cứu tiếp theo.
