"""
SmartOMR ML Grader - Module nhận dạng đáp án bằng Machine Learning
Tích hợp ý tưởng từ MLE501-ORM-Detect (standard folder)

Pipeline:
1. Nhận ảnh crop 1 câu hỏi (từ SmartOMR crop)
2. Tiền xử lý: grayscale → BW → xóa đường kẻ → tách vùng đáp án
3. Resize → 60×15 → flatten thành vector 900 features  
4. Predict bằng Logistic Regression model

Hỗ trợ 2 chế độ:
- sum_feature: dùng column-sum (60 features) → nhanh, robust hơn
- pixel_feature: dùng pixel flatten (900 features) → chi tiết hơn
"""

import cv2
import numpy as np
import os
import pickle

RESIZE_WIDTH = 60
RESIZE_HEIGHT = 15
CHOICE_LABELS = ["A", "B", "C", "D"]


# ============================================================
# TIỀN XỬ LÝ ẢNH (dựa trên MLE501 image_process.py)
# ============================================================

def convert_to_bw(img, threshold=150):
    """Chuyển ảnh sang đen trắng (inverted: bubble = trắng)."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Dùng adaptive threshold cho ảnh camera phone (chiếu sáng không đều)
    # Kết hợp cả global và adaptive
    _, global_th = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    adaptive_th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )
    
    # Lấy intersection (AND) để giảm noise
    result = cv2.bitwise_and(global_th, adaptive_th)
    return result


def remove_vertical_lines(bw_img, ratio=0.60):
    """Xóa đường kẻ dọc 2 bên (viền ô)."""
    if bw_img.size == 0:
        return bw_img
    
    col_sum = np.sum(bw_img, axis=0)
    threshold = ratio * bw_img.shape[0] * 255
    
    left_line = 0
    for i in range(len(col_sum)):
        if col_sum[i] > threshold:
            left_line = i
            break
    
    right_line = bw_img.shape[1] - 1
    for i in range(len(col_sum) - 1, -1, -1):
        if col_sum[i] > threshold:
            right_line = i
            break
    
    if right_line - left_line < 10:
        return bw_img
    
    return bw_img[:, left_line + 3:right_line - 3]


def remove_horizontal_lines(bw_img):
    """Xóa đường kẻ ngang."""
    if bw_img.size == 0:
        return bw_img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    horizontal = cv2.morphologyEx(bw_img, cv2.MORPH_OPEN, kernel)
    result = bw_img.copy()
    result[horizontal > 0] = 0
    return result


def remove_side_margin(bw_img):
    """Xóa vùng trống 2 bên."""
    if bw_img.size == 0:
        return bw_img
    col_sum = np.sum(bw_img > 0, axis=0)
    content = col_sum > 0
    if not np.any(content):
        return bw_img
    left = int(np.argmax(content))
    right = len(content) - int(np.argmax(content[::-1]))
    if right <= left:
        return bw_img
    return bw_img[:, left:right]


def extract_answer_region(bw_img):
    """
    Tách vùng đáp án (ABCD bubbles) ra khỏi vùng số câu.
    Tìm khoảng trống dọc lớn nhất giữa số câu và bubbles.
    """
    if bw_img.size == 0:
        return bw_img
    
    rmv = remove_side_margin(bw_img)
    if rmv.size == 0:
        return bw_img
    
    col_sum = np.sum(rmv > 0, axis=0)
    binary = col_sum > 0
    
    gap_start = None
    gap_len = 0
    best_gap_start = None
    best_gap_len = 0
    
    for i, v in enumerate(binary):
        if not v:
            if gap_start is None:
                gap_start = i
                gap_len = 1
            else:
                gap_len += 1
        else:
            if gap_len > best_gap_len:
                best_gap_len = gap_len
                best_gap_start = gap_start
            gap_start = None
            gap_len = 0
    
    # Chỉ cắt nếu tìm thấy gap đủ lớn (>5px sau resize)
    if best_gap_start is not None and best_gap_len > 5:
        answer_part = rmv[:, best_gap_start + best_gap_len:]
        answer_part = remove_side_margin(answer_part)
        if answer_part.size > 0:
            return answer_part
    
    return rmv


def extract_features(img, mode="raw"):
    """
    Trích xuất feature vector từ ảnh crop 1 câu hỏi.
    
    Args:
        img: ảnh BGR hoặc grayscale
        mode: "raw" (900 features, tốt nhất)
              "sum" (60 features, cũ)
              "pixel" (900 features, dùng BW preprocessing cũ)
    
    Returns:
        numpy array shape (1, n_features)
    """
    if mode == "raw":
        # Raw grayscale resize → flatten → 900 features
        # Tốt nhất cho training data chuẩn (47×364) và SmartOMR crops
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        resized = cv2.resize(gray, (RESIZE_WIDTH, RESIZE_HEIGHT))
        feature = resized.flatten().astype(np.float64) / 255.0
        return feature.reshape(1, -1)
    
    # Legacy modes (BW preprocessing)
    bw = convert_to_bw(img)
    bw = remove_vertical_lines(bw)
    bw = remove_horizontal_lines(bw)
    answer_region = extract_answer_region(bw)
    
    if answer_region.size == 0:
        if mode == "sum":
            return np.zeros((1, RESIZE_WIDTH))
        return np.zeros((1, RESIZE_WIDTH * RESIZE_HEIGHT))
    
    resized = cv2.resize(answer_region, (RESIZE_WIDTH, RESIZE_HEIGHT))
    normalized = resized / 255.0
    
    if mode == "sum":
        # Column-sum feature: tổng pixel theo cột → 60 features
        feature = np.mean(normalized, axis=0).reshape(1, -1)
    else:
        # Pixel feature: flatten toàn bộ → 900 features
        feature = normalized.flatten().reshape(1, -1)
    
    return feature


# ============================================================
# ML GRADER CLASS
# ============================================================

class MLGrader:
    """
    ML-based grader sử dụng model đã train.
    Hỗ trợ RandomForest + StandardScaler (best) hoặc LogisticRegression.
    """
    
    def __init__(self, model_path=None, mode="raw"):
        """
        Args:
            model_path: đường dẫn file .pkl chứa model (dict hoặc model trực tiếp)
            mode: "raw" (mặc định, tốt nhất) hoặc "sum" / "pixel" (legacy)
        """
        self.model = None
        self.scaler = None
        self.mode = mode
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, path):
        """Load model từ file pkl. Hỗ trợ cả format cũ và mới."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            # Format mới: {model, scaler, mode}
            self.model = data['model']
            self.scaler = data.get('scaler', None)
            saved_mode = data.get('mode', self.mode)
            if self.mode != saved_mode:
                print(f"  [ML] Mode override: {self.mode} -> {saved_mode} (từ model)")
                self.mode = saved_mode
        else:
            # Format cũ: model trực tiếp
            self.model = data
        
        print(f"  [ML] Loaded model: {path}")
        print(f"  [ML] Classes: {self.model.classes_}")
        print(f"  [ML] Mode: {self.mode}, Scaler: {'có' if self.scaler else 'không'}")
    
    def is_ready(self):
        """Kiểm tra model đã load chưa."""
        return self.model is not None
    
    def predict_one(self, img):
        """
        Predict đáp án cho 1 ảnh crop câu hỏi.
        
        Args:
            img: ảnh BGR crop 1 câu
            
        Returns:
            (label, confidence) hoặc (None, 0) nếu không chắc
        """
        if not self.is_ready():
            return None, 0.0
        
        feature = extract_features(img, mode=self.mode)
        if self.scaler is not None:
            feature = self.scaler.transform(feature)
        proba = self.model.predict_proba(feature)[0]
        pred_idx = np.argmax(proba)
        confidence = proba[pred_idx]
        label = self.model.classes_[pred_idx]
        
        return label, confidence
    
    def predict_batch(self, images):
        """
        Predict đáp án cho nhiều ảnh — vectorized (nhanh hơn ~50x so với predict_one loop).
        Trích xuất features tất cả → gọi model.predict_proba 1 lần duy nhất.
        
        Args:
            images: dict {q_num: img} hoặc list of images
            
        Returns:
            dict {q_num: (label, confidence)}
        """
        if not self.is_ready():
            return {}
        
        # Chuẩn hóa input thành ordered list
        if isinstance(images, dict):
            keys = sorted(images.keys())
            imgs = [images[k] for k in keys]
        else:
            keys = list(range(len(images)))
            imgs = list(images)
        
        if not imgs:
            return {}
        
        # Batch feature extraction (vectorized)
        features = []
        for img in imgs:
            feat = extract_features(img, mode=self.mode)  # (1, n_features)
            features.append(feat[0])
        
        X = np.array(features)  # (N, n_features)
        
        # Scale nếu có
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # GỌI MODEL 1 LẦN cho toàn bộ batch
        probas = self.model.predict_proba(X)  # (N, n_classes)
        pred_indices = np.argmax(probas, axis=1)
        confidences = probas[np.arange(len(probas)), pred_indices]
        labels = self.model.classes_[pred_indices]
        
        results = {}
        for i, key in enumerate(keys):
            results[key] = (labels[i], confidences[i])
        
        return results


# ============================================================
# TRAINING UTILITIES
# ============================================================

def prepare_training_data(data_dir, mode="sum"):
    """
    Chuẩn bị dữ liệu training từ folder chứa ảnh đã gắn nhãn.
    
    Cấu trúc folder:
        data_dir/
            A/  ← ảnh câu hỏi có đáp án A
                Q001_A.jpg
                ...
            B/
            C/
            D/
    
    Returns:
        X (features), y (labels)
    """
    X = []
    y = []
    
    # Auto-detect labels từ subfolders
    available_labels = [d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))]
    if not available_labels:
        print(f"  [ERROR] Không tìm thấy subfolder nào trong {data_dir}")
        return np.array([]), np.array([])
    
    for label in sorted(available_labels):
        folder = os.path.join(data_dir, label)
        if not os.path.isdir(folder):
            continue
        
        count = 0
        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            filepath = os.path.join(folder, filename)
            img = cv2.imread(filepath)
            if img is None:
                continue
            
            feature = extract_features(img, mode=mode)
            X.append(feature.flatten())
            y.append(label)
            count += 1
        
        print(f"  [DATA] Label {label}: {count} images")
    
    return np.array(X), np.array(y)


def train_model(data_dir, output_path="models/omr_model.pkl", mode="raw", algorithm="rf"):
    """
    Train ML model từ dữ liệu training.
    
    Args:
        data_dir: folder chứa A/, B/, C/, D/, blank/ subfolders
        output_path: đường dẫn lưu model .pkl
        mode: "raw" (mặc định, tốt nhất), "sum", "pixel"
        algorithm: "rf" (RandomForest, tốt nhất), "lr" (LogisticRegression)
    
    Returns:
        trained model
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    
    print(f"\n{'='*60}")
    print(f"  TRAINING ML MODEL")
    print(f"  Data: {data_dir}")
    print(f"  Mode: {mode}, Algorithm: {algorithm}")
    print(f"{'='*60}\n")
    
    X, y = prepare_training_data(data_dir, mode=mode)
    
    if len(X) == 0:
        print("  [ERROR] Không có dữ liệu training!")
        return None
    
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        print(f"  [ERROR] Cần ít nhất 2 class, chỉ có: {unique_labels}")
        return None
    
    print(f"\n  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Labels: {np.unique(y, return_counts=True)}")
    
    # Split train/test
    if len(X) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y
        print("  [WARN] Dataset nhỏ, dùng toàn bộ cho cả train và test")
    
    # StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if algorithm == "rf":
        model = RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1,
            max_depth=None, min_samples_split=5
        )
        # RF không cần scaling, nhưng vẫn lưu scaler cho consistency
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        train_acc = model.score(X_train, y_train)
    else:
        model = LogisticRegression(max_iter=3000, C=1.0, solver='lbfgs')
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        train_acc = model.score(X_train_scaled, y_train)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {acc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")
    
    # Save model + scaler + mode
    save_data = {
        'model': model,
        'scaler': scaler if algorithm == 'lr' else None,
        'mode': mode,
        'algorithm': algorithm,
        'accuracy': acc,
        'classes': list(model.classes_)
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"  Model saved: {output_path}")
    
    return model


def organize_training_data(questions_dir, output_dir):
    """
    Tổ chức ảnh crop từ SmartOMR thành cấu trúc training.
    SmartOMR output: Q001_A.jpg, Q002_B.jpg, ...
    Training structure: A/Q001_A.jpg, B/Q002_B.jpg, ...
    
    Args:
        questions_dir: folder chứa ảnh Q*.jpg từ SmartOMR
        output_dir: folder đích cho training data
    """
    import shutil
    
    stats = {l: 0 for l in CHOICE_LABELS}
    skipped = 0
    
    for label in CHOICE_LABELS:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)
    
    for filename in sorted(os.listdir(questions_dir)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # Parse label từ filename: Q001_A.jpg → A
        parts = filename.rsplit('.', 1)[0].split('_')
        if len(parts) < 2:
            skipped += 1
            continue
        
        label = parts[-1].upper()
        if label not in CHOICE_LABELS:
            skipped += 1
            continue
        
        src = os.path.join(questions_dir, filename)
        dst = os.path.join(output_dir, label, filename)
        shutil.copy2(src, dst)
        stats[label] += 1
    
    print(f"\n  Tổ chức training data:")
    for label, count in stats.items():
        print(f"    {label}: {count} images")
    if skipped:
        print(f"    Bỏ qua: {skipped} files")
    
    return stats


def generate_synthetic_data(questions_dir, output_dir, n_per_class=30):
    """
    Tạo dữ liệu synthetic cho B/C/D từ ảnh gốc.
    
    Phương pháp: Mỗi ảnh crop có 4 vùng bubble tương ứng A/B/C/D.
    - Tìm 4 vùng bubble qua column-sum analysis
    - Swap vùng tối (filled) sang vị trí B/C/D
    - Tạo ảnh mới với bubble khác được tô đen
    
    Args:
        questions_dir: folder chứa ảnh Q*.jpg đã crop
        output_dir: folder đích (A/, B/, C/, D/)
        n_per_class: số ảnh tối đa mỗi class
    """
    import shutil
    
    for label in CHOICE_LABELS:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)
    
    # Lấy danh sách ảnh source
    src_files = sorted([
        f for f in os.listdir(questions_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    stats = {l: 0 for l in CHOICE_LABELS}
    
    for filename in src_files:
        filepath = os.path.join(questions_dir, filename)
        img = cv2.imread(filepath)
        if img is None:
            continue
        
        # Parse label gốc
        parts = filename.rsplit('.', 1)[0].split('_')
        if len(parts) < 2:
            continue
        orig_label = parts[-1].upper()
        if orig_label not in CHOICE_LABELS:
            continue
        
        # Copy ảnh gốc vào folder đúng label
        if stats[orig_label] < n_per_class:
            dst = os.path.join(output_dir, orig_label, filename)
            shutil.copy2(filepath, dst)
            stats[orig_label] += 1
        
        # Convert sang grayscale để phân tích
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape
        
        # Tìm vùng bubbles - chia vùng phải 60% ảnh thành 4 phần
        bubble_start = int(w * 0.35)  # Bỏ qua vùng số thứ tự
        bubble_zone = gray[:, bubble_start:]
        bw = bubble_zone.shape[1]
        
        # Chia 4 vùng đều cho A/B/C/D
        quarter = bw // 4
        regions = []
        for i in range(4):
            x1 = i * quarter
            x2 = x1 + quarter if i < 3 else bw
            region = bubble_zone[:, x1:x2]
            mean_val = np.mean(region)
            regions.append((x1 + bubble_start, x2 + bubble_start, mean_val))
        
        # Tìm vùng tối nhất (bubble đã tô)
        orig_idx = CHOICE_LABELS.index(orig_label)
        filled_region = gray[:, regions[orig_idx][0]:regions[orig_idx][1]]
        
        # Tạo synthetic cho các label khác
        for target_idx, target_label in enumerate(CHOICE_LABELS):
            if target_idx == orig_idx:
                continue
            if stats[target_label] >= n_per_class:
                continue
            
            # Tạo bản copy
            synth = img.copy()
            synth_gray = gray.copy()
            
            # Bước 1: Làm sáng vùng gốc (unfill)
            orig_x1, orig_x2 = regions[orig_idx][0], regions[orig_idx][1]
            empty_mean = np.mean([regions[i][2] for i in range(4) if i != orig_idx])
            
            for c in range(synth.shape[2]) if len(synth.shape) == 3 else [0]:
                if len(synth.shape) == 3:
                    channel = synth[:, orig_x1:orig_x2, c].astype(np.float32)
                else:
                    channel = synth[:, orig_x1:orig_x2].astype(np.float32)
                
                # Brighten: scale up towards empty_mean
                filled_mean = np.mean(channel)
                if filled_mean < empty_mean and filled_mean > 10:
                    scale = min(empty_mean / filled_mean, 2.0)
                    channel = np.clip(channel * scale, 0, 255).astype(np.uint8)
                    if len(synth.shape) == 3:
                        synth[:, orig_x1:orig_x2, c] = channel
                    else:
                        synth[:, orig_x1:orig_x2] = channel
            
            # Bước 2: Tô đen vùng target (fill)
            target_x1, target_x2 = regions[target_idx][0], regions[target_idx][1]
            for c in range(synth.shape[2]) if len(synth.shape) == 3 else [0]:
                if len(synth.shape) == 3:
                    channel = synth[:, target_x1:target_x2, c].astype(np.float32)
                else:
                    channel = synth[:, target_x1:target_x2].astype(np.float32)
                
                # Darken: scale down
                channel = np.clip(channel * 0.3, 0, 255).astype(np.uint8)
                if len(synth.shape) == 3:
                    synth[:, target_x1:target_x2, c] = channel
                else:
                    synth[:, target_x1:target_x2] = channel
            
            # Lưu ảnh synthetic
            base = filename.rsplit('.', 1)[0].rsplit('_', 1)[0]
            synth_name = f"{base}_{target_label}_synth.jpg"
            cv2.imwrite(os.path.join(output_dir, target_label, synth_name), synth)
            stats[target_label] += 1
    
    print(f"\n  Synthetic data generated:")
    for label, count in stats.items():
        print(f"    {label}: {count} images")
    
    return stats


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SmartOMR ML Grader')
    sub = parser.add_subparsers(dest='command')
    
    # Command: organize
    p_org = sub.add_parser('organize', help='Tổ chức ảnh crop thành training data')
    p_org.add_argument('--input', '-i', required=True, help='Folder ảnh Q*.jpg')
    p_org.add_argument('--output', '-o', default='training_data', help='Folder output')
    
    # Command: synthetic
    p_syn = sub.add_parser('synthetic', help='Tạo synthetic training data')
    p_syn.add_argument('--input', '-i', required=True, help='Folder ảnh Q*.jpg')
    p_syn.add_argument('--output', '-o', default='training_data', help='Folder output')
    p_syn.add_argument('--n', type=int, default=60, help='Số ảnh mỗi class (default: 60)')
    
    # Command: train
    p_train = sub.add_parser('train', help='Train ML model')
    p_train.add_argument('--data', '-d', required=True, help='Folder training data (A/B/C/D/blank)')
    p_train.add_argument('--output', '-o', default='models/omr_model.pkl', help='Output model path')
    p_train.add_argument('--mode', '-m', choices=['raw', 'sum', 'pixel'], default='raw')
    p_train.add_argument('--algo', '-a', choices=['rf', 'lr'], default='rf',
                         help='Algorithm: rf=RandomForest (best), lr=LogisticRegression')
    
    # Command: predict
    p_pred = sub.add_parser('predict', help='Predict dap an tu anh')
    p_pred.add_argument('--model', required=True, help='Model .pkl path')
    p_pred.add_argument('--image', '-i', required=True, help='Anh crop 1 cau')
    
    # Command: test
    p_test = sub.add_parser('test', help='Test model tren folder anh')
    p_test.add_argument('--model', required=True, help='Model .pkl path')
    p_test.add_argument('--data', '-d', required=True, help='Folder test data')
    
    args = parser.parse_args()
    
    if args.command == 'organize':
        organize_training_data(args.input, args.output)
    
    elif args.command == 'synthetic':
        generate_synthetic_data(args.input, args.output, n_per_class=args.n)
    
    elif args.command == 'train':
        train_model(args.data, args.output, args.mode, args.algo)
    
    elif args.command == 'predict':
        grader = MLGrader(args.model)
        img = cv2.imread(args.image)
        if img is None:
            print(f"Không đọc được: {args.image}")
        else:
            label, conf = grader.predict_one(img)
            print(f"Đáp án: {label} (confidence: {conf:.4f})")
    
    elif args.command == 'test':
        grader = MLGrader(args.model)
        correct = 0
        total = 0
        # Auto-detect labels từ subfolders
        test_labels = [d for d in sorted(os.listdir(args.data))
                       if os.path.isdir(os.path.join(args.data, d))]
        for label in test_labels:
            folder = os.path.join(args.data, label)
            if not os.path.isdir(folder):
                continue
            for f in sorted(os.listdir(folder)):
                if not f.lower().endswith(('.jpg', '.png')):
                    continue
                img = cv2.imread(os.path.join(folder, f))
                if img is None:
                    continue
                pred, conf = grader.predict_one(img)
                total += 1
                if pred == label:
                    correct += 1
                else:
                    print(f"  WRONG: {f} → predicted {pred} (conf={conf:.3f}), actual {label}")
        if total > 0:
            print(f"\nAccuracy: {correct}/{total} = {correct/total:.4f}")
    
    else:
        parser.print_help()
