# SmartOMR — Nhận Dạng Phiếu Trắc Nghiệm Tự Động

> Hệ thống chấm thi trắc nghiệm tự động cho phiếu 120 câu (4 lựa chọn A/B/C/D)  
> sử dụng Computer Vision (OpenCV) với phương pháp threshold grading và Crop & Clean thông minh.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-informational)

---

## ✨ Tính Năng Nổi Bật

| Tính năng | Mô tả |
|-----------|-------|
| 🔍 **Phát hiện ô tự động** | Pipeline 6 bước dựa trên HoughCircles — hoạt động với cả ảnh scan và ảnh chụp camera |
| 📋 **Hỗ trợ 120 câu** | 4 cột × 30 hàng, đáp án A / B / C / D |
| 🎛️ **2 Chế độ Chấm Điểm** | Cung cấp 2 lựa chọn: **Standard Threshold** và **Crop & Clean Threshold** (Tẩy xóa vùng nhiễu, viền bảng) |
| 🖼️ **Gallery 120 Ảnh Giao Diện** | Cửa sổ trượt ngang (Horizontal Filmstrip) xem trực tiếp 120 ảnh câu hỏi đã cắt ngay trong App |
| 📐 **Tự động căn chỉnh góc** | Phát hiện fiducial marker và warp phối cảnh — xử lý ảnh chụp nghiêng |
| 💡 **Bù trừ ánh sáng** | CLAHE + morphology filter loại bỏ đường kẻ bảng gây nhiễu |
| 🗝️ **Đáp án linh hoạt** | Nạp từ file TXT hoặc JSON; trình soạn thảo trực quan tích hợp trong GUI |
| 📊 **Tính điểm chi tiết** | Cấu hình điểm tổng, điểm/câu đúng, trừ điểm câu sai |
| 🎨 **Annotated output** | Ảnh kết quả tô màu: xanh lá = đúng, đỏ = sai, cam = bỏ trống |
| 🔬 **Xem từng bước xử lý** | Tab "Processing Steps" hiển thị ảnh trung gian trực quan ngay trên GUI giúp dễ dàng debug (phục vụ môn học Xử lý Ảnh) |
| 🖥️ **Giao diện Desktop** | Dark-theme GUI hiện đại có tích hợp thanh cuộn mượt mà |

---

## 🔬 Phương Pháp Kỹ Thuật & Tùy Chọn Chấm (Grading Methods)

Hệ thống hiện tại cung cấp 2 phương pháp chấm độc lập hoàn toàn để so sánh độ hiệu quả:

### 1. Standard Threshold (Truyền thống)
Dựa vào nhận diện vòng tròn (HoughCircles) trên toàn bộ bức ảnh gốc kết hợp Gaussian Blur và CLAHE. Điểm sáng / tối của từng lựa chọn A, B, C, D được so sánh lẫn nhau theo **3 mức ngưỡng tương phản (Contrast Ratio)** từ chặt đến lỏng. Kèm theo đó là thuật toán **Morphology Open** ngay từ bước 3 để tự động dò vạch kẻ ngang/dọc của bảng và xóa đi, giúp bong bóng trả lời không bị dính nét đen giả.

### 2. Crop & Clean Threshold (Cắt và Làm Sạch)
Thay vì xử lý và tính điểm trên bức hình lớn, phương pháp này **nhặt ra 120 bức ảnh con (Crop)** chứa riêng 4 đáp án của từng câu. 
Trên từng bức ảnh con:
1. Áp dụng thuật toán nhị phân thích nghi để tách khối.
2. Xóa sạch viền dọc / viền ngang của bảng (Line Removal).
3. Xóa trắng hai lề trái phải (Margin Blanking) để vứt bỏ các nét dư thừa ngoài đáp án A và D.
4. Threshold lại (`OTSU`) và đếm số điểm ảnh (pixel) cực kỳ chính xác.
Đảm bảo giải quyết triệt để các hình tô mờ, lem luốc, hay vạch kẻ bảng cắt ngang ô đáp án.

---

### Sơ Đồ Pipeline (Cơ Bản)

```text
Ảnh đầu vào
    │
    ▼
[1] Tiền xử lý             ← Grayscale · Gaussian Blur · CLAHE · Xóa dòng kẻ (Morphology)
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
[5] Chấm điểm (Grading)    ← (Dùng Standard Ratio) HOẶC (Dùng Crop & Clean đếm Pixel)
    │
    ▼
[6] Xuất kết quả           ← Báo cáo điểm · Hiển thị Processing Steps · Tự nạp 120 ảnh vào Gallery
```

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
# 1. Cài đặt thư viện yêu cầu (Pillow, NumPy, OpenCV)
pip install -r requirements.txt

# 2. Khởi động giao diện GUI
python app.py

# 3. Hoặc sử dụng dòng lệnh không cần UI
python smart_omr.py -i input/sheet.jpg --answer-key answer_keys/demo_key.txt
```

---

## 📖 Hướng Dẫn Sử Dụng Giao Diện (App.py)

1. Mở App bằng `python app.py`.
2. **Chọn ảnh Input** ở cột trái (Answer Sheet).
3. Chọn file Đáp Án (Answer Key) dạng `.txt` (hoặc định nghĩa mớ bằng nút `New Key`).
4. Tại Menu **GRADING METHOD**, chọn:
   - `Standard Threshold`: Nhanh, truyền thống.
   - `Crop & Clean Threshold`: Phân tích sâu, siêu chuẩn cho ảnh bảng lưới phức tạp.
5. Xem tab **Processing Steps** bên phải để xem toàn bộ quá trình biến đổi ảnh từng bước (Grayscale -> Warp -> Lọc Vòng Tròn -> Chia cột).
6. Ở Menu EXPORT góc trái dưới cùng, nhấn **"📂 Mở thư mục 120 câu đã cắt"** để bật lên **Giao Diện Gallery Cuộn Ngang (Horizontal Filmstrip)**, cho phép xem soi từng câu trả lời đã được phần mềm cắt gọt và làm sạch. Lăn chuột để cuộn dải ảnh dẽ dàng.

---

## 📝 Định Dạng File Đáp Án (TXT)

```text
# Dòng bắt đầu bằng # là chú thích
# Định dạng: <số_câu>:<đáp_án>
1:A
2:C
3:B
...
120:D
```
Trình soạn thảo Edit Key GUI cho phép click chọn trực quan để tạo file này.

---

## 🐛 Xử Lý Sự Cố

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|------------|----------|
| "No circles found" | Ảnh quá tối hoặc nghiêng >15° | Đảm bảo đủ sáng và chụp thẳng góc |
| Thiếu thư viện Pillow | Chưa cài đặt | Chạy `pip install Pillow` |
| Bị lẹm khung kéo thanh cuộn | Màn hình độ phân giải thấp | Đã tích hợp thanh cuộn Shift+Mouseroller trong GUI |
| Căn chỉnh góc sai | Không đủ 4 marker fiducial | Đảm bảo 4 góc marker vuông đen rõ ràng không rách nét |

---

## 📄 Giấy Phép

MIT License — Tự do sử dụng, chỉnh sửa và phân phối học tập Xử Lý Ảnh.