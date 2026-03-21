# SmartOMR — Nhận Dạng Phiếu Trắc Nghiệm Tự Động

> Hệ thống chấm thi trắc nghiệm tự động cho phiếu 120 câu (4 lựa chọn A/B/C/D)  
> sử dụng Computer Vision (OpenCV) với phương pháp threshold grading.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-informational)

---

## ✨ Tính Năng

| Tính năng | Mô tả |
|-----------|-------|
| 🔍 **Phát hiện ô tự động** | Pipeline 6 bước dựa trên HoughCircles — hoạt động với cả ảnh scan và ảnh chụp camera |
| 📋 **Hỗ trợ 120 câu** | 4 cột × 30 hàng, đáp án A / B / C / D |
| 📐 **Tự động căn chỉnh góc** | Phát hiện fiducial marker và warp phối cảnh — xử lý ảnh chụp nghiêng |
| 💡 **Bù trừ ánh sáng** | CLAHE + local contrast ratio — chống lại gradient sáng từ đèn flash camera |
| 🗝️ **Đáp án linh hoạt** | Nạp từ file TXT hoặc JSON; trình soạn thảo trực quan tích hợp trong GUI |
| 📊 **Tính điểm chi tiết** | Cấu hình điểm tổng, điểm/câu đúng, trừ điểm câu sai |
| 🎨 **Annotated output** | Ảnh kết quả tô màu: xanh lá = đúng, đỏ = sai, cam = bỏ trống |
| 🔬 **Xem từng bước xử lý** | Tab "Processing Steps" hiển thị ảnh trung gian tại mỗi bước pipeline |
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
[5] Chấm điểm (Threshold)  ← Local contrast ratio + Fill ratio → 3 mức ngưỡng
    │
    ▼
[6] Xuất kết quả           ← Ảnh chú thích · Báo cáo điểm · Hiển thị GUI
```

---

### 1. Xử Lý Ảnh (Image Processing)

| Phương pháp | Hàm OpenCV | Mục đích |
|-------------|-----------|---------|
| **Chuyển đổi màu xám** | `cv2.cvtColor(BGR2GRAY)` | Giảm 3 kênh màu xuống 1 kênh cường độ sáng |
| **Gaussian Blur** | `cv2.GaussianBlur(9×9, σ=2)` | Giảm nhiễu trước khi phát hiện vòng tròn |
| **CLAHE** | `cv2.createCLAHE(clipLimit=2.0, tileGridSize=8×8)` | Cân bằng histogram thích nghi — bù trừ ánh sáng không đều từ camera |
| **Ngưỡng hóa thích nghi** | `cv2.adaptiveThreshold(GAUSSIAN_C)` | Nhị phân hóa cục bộ cho phát hiện marker |
| **Tự động co giãn** | `cv2.resize(INTER_CUBIC)` | Phóng to ảnh nhỏ lên 2500 px để đảm bảo phát hiện nhất quán |

---

### 2. Phát Hiện Đặc Trưng & Hình Học

| Phương pháp | Hàm / Công thức | Mục đích |
|-------------|----------------|---------|
| **Hough Circle Transform** | `cv2.HoughCircles(HOUGH_GRADIENT)` | Phát hiện tất cả các ô A/B/C/D trên phiếu trả lời |
| **Phát hiện đường viền** | `cv2.findContours(RETR_LIST)` | Tìm các hình khép kín để phát hiện fiducial marker |
| **Biến đổi phối cảnh** | `cv2.getPerspectiveTransform` + `cv2.warpPerspective` | Căn chỉnh ảnh chụp nghiêng về dạng nhìn từ trên xuống |
| **Lọc độ tròn** | `4π × Diện tích / Chu vi²` | Loại bỏ hình không tròn bị Hough phát hiện nhầm |
| **Lọc bán kính** | Lọc theo trung vị bán kính | Loại bỏ vòng tròn outlier quá lớn/nhỏ |

---

### 3. Phân Cụm & Phân Tích Không Gian

| Phương pháp | Mục đích |
|-------------|---------|
| **Phân cụm histogram 1D** | Nhóm các vòng tròn theo tọa độ X thành sub-column (A/B/C/D) |
| **Chia cột theo khoảng trống** | Tìm khoảng cách ngang lớn để tách thành 4 cột đáp án |
| **Nhóm hàng theo Y** | Nhóm các vòng tròn cùng vị trí dọc thành một hàng câu hỏi |
| **Dự đoán sub-column thiếu** | Nếu thiếu 1 trong 4 (A/B/C/D), dự đoán vị trí từ 3 cột còn lại |
| **Lọc theo vùng Y** | Loại bỏ vòng tròn header (mã sinh viên) ở trên cùng |

---

### 4. Phương Pháp Chấm Điểm (Threshold Grading)

Chấm điểm bằng **3 mức ngưỡng tương phản**, từ chặt → lỏng:

| Mức | Contrast Ratio | Điều kiện phụ | Trường hợp |
|-----|---------------|---------------|-----------|
| **Mức 1** | `< 0.72` | — | Tô đậm rõ ràng |
| **Mức 2** | `< 0.88` | `gap ≥ 30`, `inner_val < 180` | Tô nhạt vừa |
| **Mức 3** | `< 0.92` | `gap ≥ 40`, `fill_ratio > 0.15` | Tô rất nhạt (bút chì mờ) |

**Các chỉ số đo:**
- **Inner mean**: Độ sáng trung bình vùng lõi bubble (bán kính × 0.55)
- **Outer mean**: Độ sáng trung bình vành ngoài (nền giấy xung quanh)
- **Contrast ratio**: `inner_mean / outer_mean` — càng thấp = càng tối = càng khả thi là đã tô
- **Fill ratio**: Tỷ lệ pixel tối (< 140) trong vùng lõi — phân biệt tô thật vs nhiễu
- **Gap**: Chênh lệch inner_mean giữa bubble tối nhất và thứ 2 — đảm bảo chỉ có 1 đáp án

---

## 🚀 Hướng Dẫn Cài Đặt

### Yêu Cầu Hệ Thống

| Thành phần | Yêu cầu tối thiểu |
|-----------|------------------|
| **Python** | 3.9 trở lên |
| **RAM** | 4 GB |
| **Hệ điều hành** | Windows 10/11, macOS 11+, Ubuntu 20.04+ |

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
python smart_omr.py -i input/sheet.jpg
```

---

## 📂 Cấu Trúc Dự Án

```
SmartOMR/
├── app.py                  # Giao diện GUI (điểm vào chính)
├── smart_omr.py            # Core pipeline OMR
├── requirements.txt        # Danh sách thư viện
├── .gitignore
│
├── modules/
│   ├── __init__.py
│   └── grader.py           # Nạp đáp án, tính điểm, chú thích ảnh
│
├── answer_keys/
│   └── demo_key.txt        # File đáp án mẫu (120 câu)
│
├── input/                  # Đặt ảnh phiếu trả lời cần chấm vào đây
└── output/                 # Ảnh kết quả chú thích được lưu tại đây
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

## ⚙️ Tham Số Dòng Lệnh

| Tham số | Mặc định | Mô tả |
|--------|---------|------|
| `-i, --image` | _(bắt buộc)_ | Đường dẫn đến ảnh phiếu trả lời |
| `-k, --answer-key` | _(không có)_ | Đường dẫn đến file đáp án để chấm điểm |
| `--create-key` | — | Tạo file đáp án mẫu 120 câu |
| `-d, --debug` | `False` | Bật log chi tiết |
| `--show` | `False` | Hiển thị ảnh kết quả bằng OpenCV window |
| `--no-save` | `False` | Không lưu file kết quả |

---

## 📦 Thư Viện Sử Dụng

| Thư viện | Phiên bản | Chức năng |
|---------|----------|----------|
| `opencv-python` | ≥ 4.8 | Toàn bộ pipeline xử lý ảnh |
| `numpy` | ≥ 1.24 | Tính toán mảng số học |
| `Pillow` | ≥ 10.0 | Hiển thị ảnh trong GUI |

Cài đặt tất cả bằng lệnh: `pip install -r requirements.txt`

---

## 🐛 Xử Lý Sự Cố

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|------------|----------|
| "No circles found" | Ảnh quá tối hoặc nghiêng >15° | Đảm bảo đủ sáng và chụp thẳng góc |
| Thiếu thư viện Pillow | Chưa cài đặt | Chạy `pip install Pillow` |
| Sai số lượng câu hỏi | Phiếu không phải 120 câu | Chỉnh sửa `NUM_QUESTIONS` trong `smart_omr.py` |
| Căn chỉnh góc sai | Không đủ 4 marker fiducial | Đảm bảo 4 góc marker vuông đen rõ ràng |
| Đáp án tô mờ bị bỏ qua | Bút chì quá nhạt | Tô đậm hơn hoặc điều chỉnh threshold trong `smart_omr.py` |

---

## 📄 Giấy Phép

MIT License — Tự do sử dụng, chỉnh sửa và phân phối.