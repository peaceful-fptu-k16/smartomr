# SmartOMR — Nhận Dạng Phiếu Trắc Nghiệm Tự Động

> Hệ thống chấm thi trắc nghiệm tự động cho phiếu 120 câu (4 lựa chọn A/B/C/D)  
> sử dụng kết hợp Computer Vision và Machine Learning.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Accuracy](https://img.shields.io/badge/ML%20Accuracy-96.36%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-informational)

---

## ✨ Tính Năng

| Tính năng | Mô tả |
|-----------|-------|
| 🔍 **Phát hiện ô tự động** | Pipeline 6 bước dựa trên HoughCircles — hoạt động với cả ảnh scan và ảnh chụp camera |
| 🤖 **Chấm điểm bằng ML** | Mô hình RandomForest đạt độ chính xác **96.36%**, huấn luyện trên 30.000 mẫu |
| 📋 **Hỗ trợ 120 câu** | 4 cột × 30 hàng, đáp án A / B / C / D |
| 📐 **Tự động căn chỉnh góc** | Phát hiện fiducial marker và warp phối cảnh — xử lý ảnh chụp nghiêng |
| 💡 **Bù trừ ánh sáng** | CLAHE + local contrast ratio — chống lại gradient sáng từ đèn flash camera |
| 🗝️ **Đáp án linh hoạt** | Nạp từ file TXT hoặc JSON; trình soạn thảo trực quan tích hợp trong GUI |
| 📊 **Tính điểm chi tiết** | Cấu hình điểm tổng, điểm/câu đúng, trừ điểm câu sai |
| 🎨 **Annotated output** | Ảnh kết quả tô màu: xanh lá = đúng, đỏ = sai, cam = bỏ trống |
| 🖥️ **Giao diện Desktop** | Dark-theme GUI hiện đại — không cần web server |
| ⌨️ **Dòng lệnh (CLI)** | Xử lý hàng loạt bằng script tự động hóa |

---

## 🔬 Phương Pháp Kỹ Thuật

### Sơ Đồ Pipeline

```
Ảnh đầu vào
    │
    ▼
[1] Tiền xử lý             ← Grayscale · Gaussian Blur · CLAHE · Tự động co giãn kích thước
    │
    ▼
[2] Căn chỉnh phối cảnh    ← Phát hiện fiducial marker → warpPerspective
    │
    ▼
[3] Phát hiện ô trả lời    ← Hough Circle Transform → Lọc độ tròn → Phân cụm
    │
    ▼
[4] Xây dựng lưới câu hỏi  ← Phân cụm sub-column → Nhóm hàng → Chia cột A/B/C/D
    │
    ▼
[5] Chấm điểm              ← Local contrast ratio + ML (RandomForest) → Quyết định kết hợp
    │
    ▼
[6] Xuất kết quả           ← Ảnh chú thích · Báo cáo điểm · Hiển thị GUI
```

---

### 1. Xử Lý Ảnh (Image Processing)

| Phương pháp | Hàm OpenCV | Mục đích |
|-------------|-----------|---------|
| **Chuyển đổi màu xám** | `cv2.cvtColor(BGR2GRAY)` | Giảm 3 kênh màu xuống 1 kênh cường độ sáng |
| **Gaussian Blur** | `cv2.GaussianBlur(5×5)` | Giảm nhiễu trước khi phát hiện vòng tròn |
| **CLAHE** | `cv2.createCLAHE(clipLimit=2.0, tileGridSize=8×8)` | Cân bằng histogram thích nghi — bù trừ ánh sáng không đều từ camera |
| **Ngưỡng hóa thích nghi** | `cv2.adaptiveThreshold(GAUSSIAN_C)` | Nhị phân hóa cục bộ cho ảnh có chiếu sáng không đồng đều |
| **Ngưỡng hóa Otsu** | `cv2.threshold(THRESH_OTSU)` | Tự động chọn ngưỡng toàn cục tối ưu |
| **Phép toán hình thái** | `cv2.morphologyEx(MORPH_CLOSE/OPEN)` | Lấp lỗ hổng và nối các vùng đứt đoạn trên ô đã tô |
| **Tự động co giãn** | `cv2.resize(INTER_CUBIC)` | Phóng to ảnh nhỏ lên 2500 px để đảm bảo phát hiện nhất quán |

---

### 2. Phát Hiện Đặc Trưng & Hình Học

| Phương pháp | Hàm / Công thức | Mục đích |
|-------------|----------------|---------|
| **Hough Circle Transform** | `cv2.HoughCircles(HOUGH_GRADIENT)` | Phát hiện tất cả các ô A/B/C/D trên phiếu trả lời |
| **Phát hiện đường viền** | `cv2.findContours(RETR_LIST)` | Tìm các hình khép kín để phát hiện fiducial marker |
| **Biến đổi phối cảnh** | `cv2.getPerspectiveTransform` + `cv2.warpPerspective` | Căn chỉnh ảnh chụp nghiêng về dạng nhìn từ trên xuống |
| **Lọc độ tròn** | `4π × Diện tích / Chu vi²` | Loại bỏ hình không tròn (marker vuông, nhiễu) bị Hough phát hiện nhầm |
| **Điểm hóa góc** | `area × rectangularity × area_sum` | Chọn 4 marker góc tốt nhất tạo thành hình chữ nhật lớn nhất |
| **Phân cụm marker** | Loại bỏ theo ngưỡng khoảng cách | Gộp các detection trùng lặp của cùng một marker |

---

### 3. Phân Cụm & Phân Tích Không Gian

| Phương pháp | Mục đích |
|-------------|---------|
| **Phân cụm histogram 1D** | Nhóm các vòng tròn theo tọa độ X thành sub-column (A/B/C/D) |
| **Chia cột theo khoảng trống** | Tìm khoảng cách ngang lớn để tách thành 4 cột đáp án |
| **Nhóm hàng theo Y** | Nhóm các vòng tròn cùng vị trí dọc thành một hàng câu hỏi |
| **Dự đoán sub-column thiếu** | Nếu thiếu 1 trong 4 sub-column A/B/C/D, dự đoán vị trí từ khoảng cách 3 cột còn lại |
| **Lọc theo vùng Y** | Loại bỏ vòng tròn ngoài vùng đáp án (ví dụ: ô mã số sinh viên ở trên cùng) |
| **Phát hiện 4 góc** | Xác định 4 marker vuông lớn nhất → dùng làm điểm nguồn cho warp phối cảnh |

---

### 4. Phương Pháp Chấm Điểm

| Phương pháp | Cách hoạt động | Trường hợp áp dụng |
|-------------|---------------|-------------------|
| **Local contrast ratio** | `inner_mean / outer_mean` — so sánh độ tối bên trong ô với vòng nền xung quanh | Phương pháp chính; bền vững với gradient sáng từ camera |
| **Ngưỡng tuyệt đối** | Cường độ pixel trung bình `< 127` bên trong mặt nạ ô tròn | Dự phòng cho ảnh scan với ánh sáng đồng đều |
| **RandomForest classifier** | 900 đặc trưng pixel thô (crop 60×15) → dự đoán 5 lớp (A/B/C/D/trống) | Kiểm tra thứ cấp; ghi đè ngưỡng khi độ tin cậy > 0.85 |
| **Quyết định kết hợp** | Ngưỡng + ML đồng thuận; ML ghi đè kết quả "trống" của ngưỡng | Xác định đáp án cuối cùng |

---

### 5. Mô Hình Machine Learning

| Thuộc tính | Giá trị |
|-----------|--------|
| **Thuật toán** | `RandomForestClassifier(n_estimators=200, max_depth=20)` |
| **Đặc trưng đầu vào** | Pixel xám thô — crop ô thành 60×15 → 900 đặc trưng |
| **Chuẩn hóa** | `StandardScaler` (trung bình = 0, phương sai đơn vị) |
| **Các lớp** | `A`, `B`, `C`, `D`, `blank` (trống) |
| **Số mẫu huấn luyện** | 30.101 ảnh (≈ 6.020 mẫu/lớp) |
| **Độ chính xác test** | **96,36 %** |
| **Tỉ lệ train/test** | 80/20 phân tầng (stratified) |
| **Chế độ đặc trưng** | `raw` (pixel thô, không biến đổi thêm) |

---

## 🚀 Hướng Dẫn Cài Đặt

### Yêu Cầu Hệ Thống

| Thành phần | Yêu cầu tối thiểu |
|-----------|------------------|
| **Python** | 3.9 trở lên |
| **RAM** | 4 GB (khuyến nghị 8 GB cho xử lý hàng loạt) |
| **Hệ điều hành** | Windows 10/11, macOS 11+, Ubuntu 20.04+ |
| **Camera / Scanner** | Bất kỳ (khuyến nghị ≥ 1080p cho ảnh camera) |

### Các Bước Cài Đặt

```bash
# 1. Clone repository
git clone https://github.com/<your-username>/SmartOMR.git
cd SmartOMR

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Khởi động giao diện GUI
python app.py

# 4. Hoặc sử dụng dòng lệnh
python smart_omr.py -i input/sheet.jpg --model models/omr_model.pkl
```

---

## 📂 Cấu Trúc Dự Án

```
SmartOMR/
├── app.py                  # Giao diện GUI (điểm vào chính)
├── smart_omr.py            # Core pipeline OMR
├── train_model.py          # Script huấn luyện mô hình ML
├── requirements.txt        # Danh sách thư viện
├── .gitignore
│
├── modules/
│   ├── __init__.py
│   ├── grader.py           # Nạp đáp án, tính điểm, chú thích ảnh
│   └── ml_grader.py        # Bộ phân loại ô bằng ML + tiện ích huấn luyện
│
├── models/
│   └── omr_model.pkl       # Mô hình RandomForest đã huấn luyện (tự train)
│
├── answer_keys/
│   └── demo_key.txt        # File đáp án mẫu (120 câu)
│
├── input/                  # Đặt ảnh phiếu trả lời cần chấm vào đây
├── output/                 # Ảnh kết quả chú thích được lưu tại đây
└── training_data/          # Dữ liệu huấn luyện ML (A/B/C/D/blank/)
```

---

## 📝 Định Dạng File Đáp Án

### Định dạng TXT

```
# Dòng bắt đầu bằng # là chú thích
# Định dạng: <số_câu>:<đáp_án>
1:A
2:C
3:B
...
120:D
```

Trình soạn thảo GUI cho phép click nút radio A / B / C / D cho từng câu thay vì chỉnh sửa file thủ công.

### Định dạng JSON

```json
{
  "config": {
    "name": "Toán học cuối kỳ",
    "total_score": 10,
    "correct_score": 0.0833,
    "wrong_penalty": 0
  },
  "answers": {
    "1": "A",
    "2": "C",
    "3": "B"
  }
}
```

---

## 🧠 Huấn Luyện Mô Hình ML

Repository **không** bao gồm file mô hình đã huấn luyện (`models/omr_model.pkl`) vì kích thước lớn.  
Hãy tự huấn luyện từ ảnh ô của bạn:

```bash
# Chuẩn bị dữ liệu huấn luyện có nhãn trong thư mục training_data/<nhãn>/<ảnh>.jpg
# Nhãn: A, B, C, D, blank

python train_model.py \
    --data  training_data/ \
    --model models/omr_model.pkl \
    --mode  raw
```

Độ chính xác kỳ vọng: **≥ 95%** với 10.000+ mẫu cân bằng.

---

## ⚙️ Tham Số Dòng Lệnh

| Tham số | Mặc định | Mô tả |
|--------|---------|------|
| `-i, --image` | _(bắt buộc)_ | Đường dẫn đến ảnh phiếu trả lời |
| `-o, --output` | `output/` | Thư mục lưu kết quả đầu ra |
| `--model` | _(không có)_ | Đường dẫn đến file mô hình `.pkl` |
| `--mode` | `raw` | Chế độ trích xuất đặc trưng: `raw` / `sum` / `pixel` |
| `--answer-key` | _(không có)_ | Đường dẫn đến file đáp án để chấm điểm |
| `--create-key` | _(không có)_ | Tạo file đáp án trống tại đường dẫn chỉ định |
| `--debug` | `False` | Lưu ảnh debug trung gian |
| `--save` | `False` | Lưu ảnh kết quả chú thích |

---

## 📦 Thư Viện Sử Dụng

| Thư viện | Phiên bản | Chức năng |
|---------|----------|----------|
| `opencv-python` | ≥ 4.8 | Toàn bộ pipeline xử lý ảnh |
| `numpy` | ≥ 1.24 | Tính toán mảng số học |
| `Pillow` | ≥ 10.0 | Hiển thị ảnh trong GUI |
| `scikit-learn` | ≥ 1.3 | Bộ phân loại RandomForest |

Cài đặt tất cả bằng lệnh: `pip install -r requirements.txt`

---

## 🐛 Xử Lý Sự Cố

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|------------|----------|
| "No circles found" | Ảnh quá tối hoặc nghiêng >15° | Đảm bảo đủ sáng và chụp thẳng góc |
| Độ chính xác thấp | Dữ liệu huấn luyện không khớp máy in/scan | Huấn luyện lại mô hình với ảnh từ thiết bị của bạn |
| Thiếu thư viện Pillow | Chưa cài đặt | Chạy `pip install Pillow` |
| Sai số lượng câu hỏi | Phiếu không phải 120 câu | Chỉnh sửa `NUM_QUESTIONS` trong `smart_omr.py` |
| Căn chỉnh góc sai | Không đủ 4 marker fiducial | Đảm bảo 4 góc marker vuông đen rõ ràng trong ảnh |
| Tất cả đáp án là "blank" | Model pkl chưa được nạp | Truyền `--model models/omr_model.pkl` vào lệnh |

---

## 📄 Giấy Phép

MIT License — Tự do sử dụng, chỉnh sửa và phân phối.