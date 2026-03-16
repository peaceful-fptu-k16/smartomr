"""
SmartOMR — Script huấn luyện mô hình ML
========================================
Huấn luyện RandomForest classifier từ dữ liệu ảnh đã gắn nhãn.

Cách dùng:
    python train_model.py --data training_data/ --model models/omr_model.pkl --mode raw

Cấu trúc thư mục dữ liệu:
    training_data/
        A/      ← ảnh ô có đáp án A
        B/
        C/
        D/
        blank/  ← ảnh ô trống
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ml_grader import train_model


def main():
    parser = argparse.ArgumentParser(
        description="SmartOMR — Huấn luyện mô hình ML nhận dạng đáp án"
    )
    parser.add_argument(
        "--data", "-d", required=True,
        help="Thư mục chứa dữ liệu huấn luyện (A/B/C/D/blank)"
    )
    parser.add_argument(
        "--model", "-o", default="models/omr_model.pkl",
        help="Đường dẫn lưu model (mặc định: models/omr_model.pkl)"
    )
    parser.add_argument(
        "--mode", "-m", choices=["raw", "sum", "pixel"], default="raw",
        help="Chế độ trích xuất đặc trưng (mặc định: raw)"
    )
    parser.add_argument(
        "--algo", "-a", choices=["rf", "lr"], default="rf",
        help="Thuật toán: rf=RandomForest (tốt nhất), lr=LogisticRegression"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.data):
        print(f"[ERROR] Không tìm thấy thư mục: {args.data}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.model) or ".", exist_ok=True)
    train_model(args.data, args.model, args.mode, args.algo)


if __name__ == "__main__":
    main()
