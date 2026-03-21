"""
SmartOMR v3 - Hệ thống nhận dạng phiếu thi trắc nghiệm 
Phiếu chuẩn tnmaker.net, 120 câu (4 cột × 30 câu × 4 lựa chọn ABCD)

Tiếp cận:
1. HoughCircles tìm tất cả ô tròn
2. Lọc theo radius & vị trí, bỏ header
3. Cluster X → 4 cột chính → trong mỗi cột 4 sub-col (A,B,C,D)  
4. Cluster Y → 30 hàng (câu hỏi)
5. Grading bằng mean grayscale (không dùng adaptive threshold)
6. Cắt ảnh từng câu: "Số câu | O O O O"
"""

import cv2
import numpy as np
import os
import sys
import time



try:
    from modules.grader import load_answer_key, grade, draw_graded_annotated
    GRADER_AVAILABLE = True
except ImportError:
    GRADER_AVAILABLE = False


# ============================================================
# CẤU HÌNH
# ============================================================
NUM_QUESTIONS = 120
NUM_CHOICES = 4
NUM_COLUMNS = 4
QUESTIONS_PER_COLUMN = 30
CHOICE_LABELS = ["A", "B", "C", "D"]

# HoughCircles
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 30
HOUGH_PARAM1 = 50
HOUGH_PARAM2 = 30
HOUGH_MIN_RADIUS = 15
HOUGH_MAX_RADIUS = 40

# Grading: dùng mean grayscale (0=đen, 255=trắng)
# Ô tô: mean ~40-120, Ô trống: mean ~190-220
FILLED_MEAN_THRESHOLD = 160   # mean < 160 → filled
EMPTY_MEAN_THRESHOLD = 180    # mean > 180 → chắc chắn empty

# Crop padding
CROP_PAD_X = 25
CROP_PAD_Y = 8

# Kích thước chuẩn để tự co giãn ảnh trước khi nhận diện.
AUTO_SCALE_TARGET_W = 2500


class SmartOMR:
    def __init__(self, debug=False):
        """Khởi tạo bộ xử lý OMR.

        Args:
            debug: Bật log chi tiết khi xử lý.
        """
        self.debug = debug
        self._gray_clahe_cache = None
        self._step_images = []

    def _add_step(self, name, desc, img):
        """Lưu ảnh trung gian của 1 bước xử lý."""
        if len(img.shape) == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis = img.copy()
        self._step_images.append((name, desc, vis))

    def process(self, image_path):
        """Xử lý 1 ảnh phiếu và trả về đáp án + ảnh annotate + thống kê."""
        self._step_images = []  # reset
        print(f"\n{'='*60}")
        print(f"  SmartOMR v3")
        print(f"  Ảnh: {image_path}")
        print(f"{'='*60}")

        t0 = time.time()
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Không đọc được: {image_path}")
            return None

        # === STEP 1: Ảnh gốc ===
        self._add_step(
            "Ảnh gốc (Original)",
            f"Ảnh đầu vào chưa xử lý.\nKích thước: {image.shape[1]}x{image.shape[0]}",
            image)

        # Tự co giãn ảnh đầu vào để ổn định bước HoughCircles.
        h_orig, w_orig = image.shape[:2]
        self._scale_factor = 1.0
        if w_orig < AUTO_SCALE_TARGET_W * 0.75:
            self._scale_factor = AUTO_SCALE_TARGET_W / w_orig
            new_w = AUTO_SCALE_TARGET_W
            new_h = int(h_orig * self._scale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            print(f"  Auto-scaled: {w_orig}x{h_orig} → {new_w}x{new_h} (×{self._scale_factor:.2f})")

        # === STEP 2: Grayscale ===
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        print(f"  Kích thước: {w}x{h}")
        self._add_step(
            "Grayscale",
            "Chuyển ảnh màu sang thang xám (0–255).\n"
            "Công thức: Y = 0.299*R + 0.587*G + 0.114*B",
            gray)

        # Hiệu chỉnh phối cảnh dựa trên marker 4 góc.
        warped = self._perspective_correct(image, gray)
        if warped is not None:
            image = warped
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            print(f"  Perspective corrected: {w}x{h}")
            self._add_step(
                "Perspective Warp",
                "Hiệu chỉnh phối cảnh dựa trên maker 4 góc.\n"
                "Tìm marker → getPerspectiveTransform → warpPerspective",
                image)

        # Co giãn lại sau warp về kích thước chuẩn để pipeline ổn định.
        h_cur, w_cur = image.shape[:2]
        if w_cur < AUTO_SCALE_TARGET_W * 0.90 or w_cur > AUTO_SCALE_TARGET_W * 1.10:
            scale = AUTO_SCALE_TARGET_W / w_cur
            new_w = AUTO_SCALE_TARGET_W
            new_h = int(h_cur * scale)
            interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            image = cv2.resize(image, (new_w, new_h), interpolation=interp)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            self._scale_factor *= scale
            print(f"  Post-warp scale: {w_cur}x{h_cur} -> {new_w}x{new_h} (x{scale:.2f})")

        # === STEP 3: CLAHE ===
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._gray_clahe_cache = clahe.apply(gray)
        self._add_step(
            "CLAHE (Tăng tương phản)",
            "Contrast Limited Adaptive Histogram Equalization.\n"
            "Tăng tương phản cục bộ giúp nhận diện bubble tốt hơn.\n"
            f"clipLimit=2.0, tileGridSize=(8,8)",
            self._gray_clahe_cache)

        # === STEP 4: GaussianBlur ===
        enhanced = self._gray_clahe_cache
        blurred = cv2.GaussianBlur(enhanced, (9, 9), 2)
        self._add_step(
            "GaussianBlur",
            "Làm mờ Gaussian trước khi phát hiện vòng tròn.\n"
            "Giảm nhiễu để HoughCircles ổn định hơn.\n"
            "kernel=(9,9), sigma=2",
            blurred)

        # === BƯỚC 1: Phát hiện tất cả circles ===
        print(f"\n[1/6] HoughCircles...")
        raw_circles = self._detect_circles(gray)
        print(f"  -> {len(raw_circles)} circles")

        # === STEP 5: HoughCircles (raw) ===
        vis_raw = image.copy()
        for (cx, cy, r) in raw_circles:
            cv2.circle(vis_raw, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(vis_raw, (cx, cy), 2, (0, 0, 255), 3)
        self._add_step(
            "HoughCircles (Raw)",
            f"Phát hiện vòng tròn bằng HoughCircles.\n"
            f"Tổng số: {len(raw_circles)} circles.\n"
            f"dp={HOUGH_DP}, minDist={HOUGH_MIN_DIST}, "
            f"param1={HOUGH_PARAM1}, param2={HOUGH_PARAM2}",
            vis_raw)

        # === BƯỚC 2: Lọc radius, bỏ outlier ===
        print(f"[2/6] Lọc theo radius...")
        good_circles = self._filter_by_radius(raw_circles, gray)
        print(f"  -> {len(good_circles)} circles (radius lọc)")

        # === STEP 6: Lọc Radius & Circularity ===
        vis_filt = image.copy()
        for (cx, cy, r) in good_circles:
            cv2.circle(vis_filt, (cx, cy), r, (255, 180, 0), 2)
            cv2.circle(vis_filt, (cx, cy), 2, (0, 0, 255), 3)
        self._add_step(
            "Lọc Radius & Circularity",
            f"Lọc bỏ vòng tròn outlier theo bán kính trung vị\n"
            f"và độ tròn (circularity).\n"
            f"Còn lại: {len(good_circles)}/{len(raw_circles)} circles",
            vis_filt)

        # === BƯỚC 3: Phân 4 cột chính theo X ===
        print(f"[3/6] Chia 4 cột chính...")
        columns = self._split_into_main_columns(good_circles)
        for ci, col in enumerate(columns):
            if col:
                xs = [c[0] for c in col]
                print(f"  Cột {ci+1}: {len(col)} circles, x=[{min(xs)},{max(xs)}]")

        # === STEP 7: Chia 4 cột ===
        col_colors = [(0, 0, 255), (0, 200, 0), (255, 100, 0), (200, 0, 200)]
        vis_cols = image.copy()
        col_desc = ""
        for ci, col in enumerate(columns):
            color = col_colors[ci % 4]
            for (cx, cy, r) in col:
                cv2.circle(vis_cols, (cx, cy), r, color, 2)
            if col:
                xs = [c[0] for c in col]
                col_desc += f"Cột {ci+1}: {len(col)} circles  "
        self._add_step(
            "Chia 4 cột chính",
            f"Phân nhóm circles theo tọa độ X thành 4 cột.\n"
            f"Tìm 3 khoảng trống lớn nhất giữa các sub-column.\n"
            f"{col_desc}",
            vis_cols)

        # === BƯỚC 4: Trong mỗi cột, phân hàng + ABCD ===
        print(f"[4/6] Phân hàng & xác định A/B/C/D...")
        grid = self._build_answer_grid(columns, gray)
        total_q = sum(len(rows) for rows in grid.values())
        print(f"  -> {total_q} câu nhận diện")

        # === STEP 8: Answer Grid ===
        vis_grid = image.copy()
        q_num = 1
        abcd_colors = [(0, 0, 220), (0, 180, 0), (220, 120, 0), (180, 0, 180)]
        for col_idx in sorted(grid.keys()):
            for row in grid[col_idx]:
                if q_num > NUM_QUESTIONS:
                    break
                for ci, (cx, cy, r) in enumerate(row[:NUM_CHOICES]):
                    cv2.circle(vis_grid, (cx, cy), r, abcd_colors[ci % 4], 2)
                if row:
                    cv2.putText(vis_grid, str(q_num),
                                (row[0][0] - row[0][2] - 50, row[0][1] + 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                q_num += 1
        self._add_step(
            "Answer Grid (Hàng & ABCD)",
            f"Phân hàng (cluster Y) và xác định vị trí A/B/C/D.\n"
            f"Bỏ header row, giữ 30 hàng/cột.\n"
            f"Tổng: {total_q} câu nhận diện.\n"
            f"Màu: Đỏ=A, Xanh=B, Cam=C, Tím=D",
            vis_grid)

        # === BƯỚC 5: Grading (threshold) ===
        print(f"[5/6] Xác định đáp án (threshold)...")
        answers = self._grade_all(gray, grid)
        n_answered = sum(1 for v in answers.values() if v is not None)
        print(f"  -> {n_answered}/{len(answers)} câu có đáp án")

        # === STEP 9: Threshold Grading ===
        vis_thresh = image.copy()
        q_num = 1
        for col_idx in sorted(grid.keys()):
            for row in grid[col_idx]:
                if q_num > NUM_QUESTIONS:
                    break
                answer = answers.get(q_num)
                for ci, (cx, cy, r) in enumerate(row[:NUM_CHOICES]):
                    if answer and ci < len(CHOICE_LABELS) and CHOICE_LABELS[ci] == answer:
                        cv2.circle(vis_thresh, (cx, cy), r + 5, (0, 0, 255), 3)
                        cv2.circle(vis_thresh, (cx, cy), r, (0, 0, 255), -1)
                    else:
                        cv2.circle(vis_thresh, (cx, cy), r, (0, 200, 0), 1)
                if row:
                    label = f"{q_num}:{answer or '-'}"
                    cv2.putText(vis_thresh, label,
                                (row[0][0] - row[0][2] - 70, row[0][1] + 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
                q_num += 1
        self._add_step(
            "Threshold Grading",
            f"Xác định đáp án bằng phân tích độ sáng vùng lõi.\n"
            f"Bubble tối nhất + contrast với vùng xung quanh.\n"
            f"Kết quả: {n_answered}/{len(answers)} câu có đáp án",
            vis_thresh)

        # === BƯỚC 6: Cắt ảnh từng câu ===
        print(f"[6/6] Cắt ảnh câu hỏi...")
        question_images = self._crop_all_questions(image, gray, grid, answers)
        print(f"  -> {len(question_images)} ảnh")

        # === STEP 10: Kết quả cuối cùng ===
        annotated = self._draw_annotated(image, grid, answers)
        self._add_step(
            "Kết quả Annotated",
            f"Ảnh kết quả cuối cùng với vòng tròn đáp án\n"
            f"và số thứ tự câu hỏi.",
            annotated)

        elapsed = time.time() - t0

        stats = {
            'total_circles': len(raw_circles),
            'filtered_circles': len(good_circles),
            'total_questions': NUM_QUESTIONS,
            'detected_questions': total_q,
            'answered': n_answered,
            'unanswered': total_q - n_answered,
            'processing_time': elapsed
        }

        self._print_results(answers, stats)

        # Giải phóng cache (tránh memory leak khi xử lý nhiều ảnh liên tiếp)
        self._gray_clahe_cache = None

        return {
            'answers': answers,
            'annotated': annotated,
            'question_images': question_images,
            'stats': stats,
            'grid': grid,
            'image_orig': image,
            'step_images': list(self._step_images)
        }

    # ===================================================================
    # HIỆU CHỈNH PHỐI CẢNH DỰA TRÊN MARKER GÓC
    # ===================================================================
    def _perspective_correct(self, image, gray):
        """Phát hiện marker 4 góc và warp ảnh về khung chuẩn.

        Trả về:
            Ảnh đã warp hoặc None nếu không đủ điều kiện.
        """
        h, w = gray.shape
        
        # Adaptive threshold để tìm marker ổn định trên ảnh ánh sáng không đều.
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 21, 10)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Lọc marker dạng vuông/tối (gần giống ký hiệu góc của form).
        min_side = w * 0.008
        max_side = w * 0.04
        min_area = min_side ** 2
        max_area = max_side ** 2
        
        markers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / bh if bh > 0 else 0
            fill = area / (bw * bh) if bw * bh > 0 else 0
            if 0.6 < aspect < 1.7 and fill > 0.6:
                roi = gray[y:y+bh, x:x+bw]
                if roi.mean() < 140:
                    cx, cy = x + bw // 2, y + bh // 2
                    markers.append((cx, cy, area))
        
        if len(markers) < 3:
            return None
        
        # Gom marker gần nhau (tránh trùng một marker do detect nhiều lần).
        clustered = self._cluster_markers(markers)
        
        if len(clustered) < 3:
            return None
        
        # Tìm 4 marker góc tốt nhất.
        corners = self._find_corner_markers(clustered, w, h)
        
        # Nếu thiếu 1 góc thì ước lượng từ 3 góc (quy tắc hình bình hành).
        if corners is None and len(clustered) >= 3:
            corners = self._estimate_4th_corner(clustered, w, h)
        
        if corners is None:
            return None
        
        tl, tr, bl, br = corners
        
        # Kiểm tra tứ giác hợp lệ trước khi warp.
        quad_w = max(np.linalg.norm(np.array(tr) - np.array(tl)),
                     np.linalg.norm(np.array(br) - np.array(bl)))
        quad_h = max(np.linalg.norm(np.array(bl) - np.array(tl)),
                     np.linalg.norm(np.array(br) - np.array(tr)))
        
        # Tứ giác phải đủ lớn để tránh warp nhầm vùng nhiễu.
        if quad_w < w * 0.35 or quad_h < h * 0.35:
            return None
        
        # Khung đích chuẩn, có padding để không cắt sát mép.
        pad = 20
        dst_w = int(quad_w) + 2 * pad
        dst_h = int(quad_h) + 2 * pad
        
        src = np.array([tl, tr, br, bl], dtype=np.float32)
        dst = np.array([
            [pad, pad],
            [dst_w - pad, pad],
            [dst_w - pad, dst_h - pad],
            [pad, dst_h - pad]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, M, (dst_w, dst_h),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)
        
        print(f"  Markers detected: TL({tl[0]:.0f},{tl[1]:.0f}) "
              f"TR({tr[0]:.0f},{tr[1]:.0f}) "
              f"BL({bl[0]:.0f},{bl[1]:.0f}) "
              f"BR({br[0]:.0f},{br[1]:.0f})")
        
        return warped
    
    def _cluster_markers(self, markers, dist_threshold=30):
        """Gộp các marker quá gần nhau thành 1 marker đại diện.

        Args:
            markers: Danh sách marker (cx, cy, area).
            dist_threshold: Ngưỡng khoảng cách để coi là trùng marker.
        """
        markers = sorted(markers, key=lambda m: m[2], reverse=True)  # ưu tiên marker có area lớn
        used = [False] * len(markers)
        clustered = []
        
        for i, (cx, cy, area) in enumerate(markers):
            if used[i]:
                continue
            used[i] = True 
            # Đánh dấu marker lân cận là đã dùng.
            for j in range(i + 1, len(markers)):
                if not used[j]:
                    dx = abs(markers[j][0] - cx)
                    dy = abs(markers[j][1] - cy)
                    if dx < dist_threshold and dy < dist_threshold:
                        used[j] = True
            clustered.append((cx, cy, area))
        
        return clustered
    
    def _find_corner_markers(self, markers, img_w, img_h):
        """Chọn 4 marker tạo thành góc của vùng làm bài.

        Chiến lược: lấy các marker lớn nhất, thử tổ hợp 4 điểm và chấm điểm
        theo độ "hình chữ nhật" + diện tích.
        """
        # Marker góc thường có diện tích lớn nhất trên phiếu.
        markers_sorted = sorted(markers, key=lambda m: m[2], reverse=True)
        
        # Chỉ lấy top candidate để giảm độ phức tạp tổ hợp.
        candidates = markers_sorted[:min(10, len(markers_sorted))]
        
        # Thử các tổ hợp 4 điểm và chọn tổ hợp có score tốt nhất.
        from itertools import combinations
        
        best_score = -1
        best_corners = None
        
        for combo in combinations(range(len(candidates)), 4):
            pts = [(candidates[i][0], candidates[i][1]) for i in combo]
            
            tl = min(pts, key=lambda p: p[0] + p[1])
            tr = max(pts, key=lambda p: p[0] - p[1])
            br = max(pts, key=lambda p: p[0] + p[1])
            bl = min(pts, key=lambda p: p[0] - p[1])
            
            if len({tl, tr, br, bl}) < 4:
                continue
            
            # Cặp trên và cặp dưới cần có Y gần nhau.
            top_dy = abs(tl[1] - tr[1])
            bot_dy = abs(bl[1] - br[1])
            if top_dy > img_h * 0.15 or bot_dy > img_h * 0.15:
                continue
            
            # Kích thước khung phải hợp lý.
            w1 = np.linalg.norm(np.array(tr) - np.array(tl))
            w2 = np.linalg.norm(np.array(br) - np.array(bl))
            h1 = np.linalg.norm(np.array(bl) - np.array(tl))
            h2 = np.linalg.norm(np.array(br) - np.array(tr))
            
            if min(w1, w2) < img_w * 0.3 or min(h1, h2) < img_h * 0.3:
                continue
            
            # Score ưu tiên: diện tích lớn + hình chữ nhật rõ.
            area = 0.5 * abs(
                (tr[0] - tl[0]) * (br[1] - tl[1]) - (br[0] - tl[0]) * (tr[1] - tl[1]) +
                (br[0] - tr[0]) * (bl[1] - tr[1]) - (bl[0] - tr[0]) * (br[1] - tr[1])
            )
            w_ratio = min(w1, w2) / max(w1, w2) if max(w1, w2) > 0 else 0
            h_ratio = min(h1, h2) / max(h1, h2) if max(h1, h2) > 0 else 0
            rectangularity = w_ratio * h_ratio  # càng gần 1 càng vuông vức
            
            # Ưu tiên thêm các combo có marker lớn.
            area_sum = sum(candidates[i][2] for i in combo)
            
            score = area * rectangularity * area_sum
            
            if score > best_score:
                best_score = score
                best_corners = (tl, tr, bl, br)
        
        return best_corners

    def _estimate_4th_corner(self, markers, img_w, img_h):
        """Ước lượng góc thứ 4 khi chỉ tìm được 3 marker góc."""
        from itertools import combinations
        
        markers_sorted = sorted(markers, key=lambda m: m[2], reverse=True)
        candidates = markers_sorted[:min(10, len(markers_sorted))]
        
        best_score = -1
        best_corners = None
        
        for combo in combinations(range(len(candidates)), 3):
            pts = [(candidates[i][0], candidates[i][1]) for i in combo]
            
            # Thử lần lượt từng vai trò góc bị thiếu.
            for missing_role in ['tl', 'tr', 'bl', 'br']:
                # Gán 3 điểm hiện có vào các vai trò còn lại.
                if missing_role == 'bl':
                    # Have TL, TR, BR → BL = TL + BR - TR
                    tl = min(pts, key=lambda p: p[0] + p[1])
                    tr = max(pts, key=lambda p: p[0] - p[1])
                    br = max(pts, key=lambda p: p[0] + p[1])
                    if len({tl, tr, br}) < 3:
                        continue
                    bl = (tl[0] + br[0] - tr[0], tl[1] + br[1] - tr[1])
                elif missing_role == 'br':
                    tl = min(pts, key=lambda p: p[0] + p[1])
                    tr = max(pts, key=lambda p: p[0] - p[1])
                    bl = min(pts, key=lambda p: p[0] - p[1])
                    if len({tl, tr, bl}) < 3:
                        continue
                    br = (tr[0] + bl[0] - tl[0], tr[1] + bl[1] - tl[1])
                elif missing_role == 'tl':
                    tr = max(pts, key=lambda p: p[0] - p[1])
                    bl = min(pts, key=lambda p: p[0] - p[1])
                    br = max(pts, key=lambda p: p[0] + p[1])
                    if len({tr, bl, br}) < 3:
                        continue
                    tl = (tr[0] + bl[0] - br[0], tr[1] + bl[1] - br[1])
                else:  # missing_role == 'tr'
                    tl = min(pts, key=lambda p: p[0] + p[1])
                    bl = min(pts, key=lambda p: p[0] - p[1])
                    br = max(pts, key=lambda p: p[0] + p[1])
                    if len({tl, bl, br}) < 3:
                        continue
                    tr = (tl[0] + br[0] - bl[0], tl[1] + br[1] - bl[1])
                
                # Góc ước lượng phải nằm trong biên ảnh (cho phép margin nhỏ).
                for p in [tl, tr, bl, br]:
                    if p[0] < -img_w * 0.1 or p[0] > img_w * 1.1:
                        break
                    if p[1] < -img_h * 0.1 or p[1] > img_h * 1.1:
                        break
                else:
                    # Kiểm tra kích thước khung sau ước lượng.
                    w1 = np.linalg.norm(np.array(tr) - np.array(tl))
                    w2 = np.linalg.norm(np.array(br) - np.array(bl))
                    h1 = np.linalg.norm(np.array(bl) - np.array(tl))
                    h2 = np.linalg.norm(np.array(br) - np.array(tr))
                    
                    if min(w1, w2) < img_w * 0.3 or min(h1, h2) < img_h * 0.3:
                        continue
                    
                    top_dy = abs(tl[1] - tr[1])
                    bot_dy = abs(bl[1] - br[1])
                    if top_dy > img_h * 0.15 or bot_dy > img_h * 0.15:
                        continue
                    
                    w_ratio = min(w1, w2) / max(w1, w2) if max(w1, w2) > 0 else 0
                    h_ratio = min(h1, h2) / max(h1, h2) if max(h1, h2) > 0 else 0
                    rectangularity = w_ratio * h_ratio
                    
                    area_sum = sum(candidates[i][2] for i in combo)
                    area = w1 * h1
                    score = area * rectangularity * area_sum
                    
                    if score > best_score:
                        best_score = score
                        best_corners = (tl, tr, bl, br)
        
        return best_corners

    # ===================================================================
    # BƯỚC 1: PHÁT HIỆN CIRCLE
    # ===================================================================
    def process_crop_clean(self, image_path):
        """Xử lý 1 ảnh phiếu theo phương pháp tách biệt: Crop & Clean."""
        self._step_images = []  # reset
        print(f"\n{'='*60}")
        print(f"  SmartOMR v3 (Crop & Clean Method)")
        print(f"  Ảnh: {image_path}")
        print(f"{'='*60}")

        t0 = time.time()
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Không đọc được: {image_path}")
            return None

        # === STEP 1: Ảnh gốc ===
        self._add_step(
            "Ảnh gốc (Original)",
            f"Ảnh đầu vào chưa xử lý.\nKích thước: {image.shape[1]}x{image.shape[0]}",
            image)

        # Tự co giãn ảnh đầu vào để ổn định bước HoughCircles.
        h_orig, w_orig = image.shape[:2]
        self._scale_factor = 1.0
        if w_orig < AUTO_SCALE_TARGET_W * 0.75:
            self._scale_factor = AUTO_SCALE_TARGET_W / w_orig
            new_w = AUTO_SCALE_TARGET_W
            new_h = int(h_orig * self._scale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            print(f"  Auto-scaled: {w_orig}x{h_orig} → {new_w}x{new_h} (×{self._scale_factor:.2f})")

        # === STEP 2: Grayscale ===
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        print(f"  Kích thước: {w}x{h}")
        self._add_step(
            "Grayscale",
            "Chuyển ảnh màu sang thang xám (0–255).\n"
            "Công thức: Y = 0.299*R + 0.587*G + 0.114*B",
            gray)

        # Hiệu chỉnh phối cảnh dựa trên marker 4 góc.
        warped = self._perspective_correct(image, gray)
        if warped is not None:
            image = warped
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            print(f"  Perspective corrected: {w}x{h}")
            self._add_step(
                "Perspective Warp",
                "Hiệu chỉnh phối cảnh dựa trên maker 4 góc.\n"
                "Tìm marker → getPerspectiveTransform → warpPerspective",
                image)

        # Co giãn lại sau warp về kích thước chuẩn để pipeline ổn định.
        h_cur, w_cur = image.shape[:2]
        if w_cur < AUTO_SCALE_TARGET_W * 0.90 or w_cur > AUTO_SCALE_TARGET_W * 1.10:
            scale = AUTO_SCALE_TARGET_W / w_cur
            new_w = AUTO_SCALE_TARGET_W
            new_h = int(h_cur * scale)
            interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            image = cv2.resize(image, (new_w, new_h), interpolation=interp)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            self._scale_factor *= scale
            print(f"  Post-warp scale: {w_cur}x{h_cur} -> {new_w}x{new_h} (x{scale:.2f})")

        # === STEP 3: CLAHE ===
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._gray_clahe_cache = clahe.apply(gray)
        self._add_step(
            "CLAHE (Tăng tương phản)",
            "Contrast Limited Adaptive Histogram Equalization.\n"
            "Tăng tương phản cục bộ giúp nhận diện bubble tốt hơn.\n"
            f"clipLimit=2.0, tileGridSize=(8,8)",
            self._gray_clahe_cache)

        # === STEP 4: GaussianBlur ===
        enhanced = self._gray_clahe_cache
        blurred = cv2.GaussianBlur(enhanced, (9, 9), 2)
        self._add_step(
            "GaussianBlur",
            "Làm mờ Gaussian trước khi phát hiện vòng tròn.\n"
            "Giảm nhiễu để HoughCircles ổn định hơn.\n"
            "kernel=(9,9), sigma=2",
            blurred)

        # === BƯỚC 1: Phát hiện tất cả circles ===
        print(f"\n[1/6] HoughCircles...")
        raw_circles = self._detect_circles(gray)
        print(f"  -> {len(raw_circles)} circles")

        # === STEP 5: HoughCircles (raw) ===
        vis_raw = image.copy()
        for (cx, cy, r) in raw_circles:
            cv2.circle(vis_raw, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(vis_raw, (cx, cy), 2, (0, 0, 255), 3)
        self._add_step(
            "HoughCircles (Raw)",
            f"Phát hiện vòng tròn bằng HoughCircles.\n"
            f"Tổng số: {len(raw_circles)} circles.\n"
            f"dp={HOUGH_DP}, minDist={HOUGH_MIN_DIST}, "
            f"param1={HOUGH_PARAM1}, param2={HOUGH_PARAM2}",
            vis_raw)

        # === BƯỚC 2: Lọc radius, bỏ outlier ===
        print(f"[2/6] Lọc theo radius...")
        good_circles = self._filter_by_radius(raw_circles, gray)
        print(f"  -> {len(good_circles)} circles (radius lọc)")

        # === STEP 6: Lọc Radius & Circularity ===
        vis_filt = image.copy()
        for (cx, cy, r) in good_circles:
            cv2.circle(vis_filt, (cx, cy), r, (255, 180, 0), 2)
            cv2.circle(vis_filt, (cx, cy), 2, (0, 0, 255), 3)
        self._add_step(
            "Lọc Radius & Circularity",
            f"Lọc bỏ vòng tròn outlier theo bán kính trung vị\n"
            f"và độ tròn (circularity).\n"
            f"Còn lại: {len(good_circles)}/{len(raw_circles)} circles",
            vis_filt)

        # === BƯỚC 3: Phân 4 cột chính theo X ===
        print(f"[3/6] Chia 4 cột chính...")
        columns = self._split_into_main_columns(good_circles)
        for ci, col in enumerate(columns):
            if col:
                xs = [c[0] for c in col]
                print(f"  Cột {ci+1}: {len(col)} circles, x=[{min(xs)},{max(xs)}]")

        # === STEP 7: Chia 4 cột ===
        col_colors = [(0, 0, 255), (0, 200, 0), (255, 100, 0), (200, 0, 200)]
        vis_cols = image.copy()
        col_desc = ""
        for ci, col in enumerate(columns):
            color = col_colors[ci % 4]
            for (cx, cy, r) in col:
                cv2.circle(vis_cols, (cx, cy), r, color, 2)
            if col:
                xs = [c[0] for c in col]
                col_desc += f"Cột {ci+1}: {len(col)} circles  "
        self._add_step(
            "Chia 4 cột chính",
            f"Phân nhóm circles theo tọa độ X thành 4 cột.\n"
            f"Tìm 3 khoảng trống lớn nhất giữa các sub-column.\n"
            f"{col_desc}",
            vis_cols)

        # === BƯỚC 4: Trong mỗi cột, phân hàng + ABCD ===
        print(f"[4/6] Phân hàng & xác định A/B/C/D...")
        grid = self._build_answer_grid(columns, gray)
        total_q = sum(len(rows) for rows in grid.values())
        print(f"  -> {total_q} câu nhận diện")

        # === STEP 8: Answer Grid ===
        vis_grid = image.copy()
        q_num = 1
        abcd_colors = [(0, 0, 220), (0, 180, 0), (220, 120, 0), (180, 0, 180)]
        for col_idx in sorted(grid.keys()):
            for row in grid[col_idx]:
                if q_num > NUM_QUESTIONS:
                    break
                for ci, (cx, cy, r) in enumerate(row[:NUM_CHOICES]):
                    cv2.circle(vis_grid, (cx, cy), r, abcd_colors[ci % 4], 2)
                if row:
                    cv2.putText(vis_grid, str(q_num),
                                (row[0][0] - row[0][2] - 50, row[0][1] + 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                q_num += 1
        self._add_step(
            "Answer Grid (Hàng & ABCD)",
            f"Phân hàng (cluster Y) và xác định vị trí A/B/C/D.\n"
            f"Bỏ header row, giữ 30 hàng/cột.\n"
            f"Tổng: {total_q} câu nhận diện.\n"
            f"Màu: Đỏ=A, Xanh=B, Cam=C, Tím=D",
            vis_grid)

        # === BƯỚC 5: Crop & Clean & Grade ===
        print(f"[5/6] Cắt ảnh và chấm Crop & Clean...")

        answers = {}
        question_images = {}
        q_num = 1
        
        vis_thresh = image.copy()

        for col_idx in sorted(grid.keys()):
            for row in grid[col_idx]:
                if q_num > NUM_QUESTIONS:
                    break
                
                n = min(len(row), NUM_CHOICES)
                if n < 3:
                    answers[q_num] = None
                    q_num += 1
                    continue
                
                # Sắp xếp bubbles từ A -> D
                row_sorted = sorted(row[:n], key=lambda c: c[0])
                
                # Bounding box của dòng này
                min_x = int(min(c[0] - c[2] for c in row_sorted))
                max_x = int(max(c[0] + c[2] for c in row_sorted))
                min_y = int(min(c[1] - c[2] for c in row_sorted))
                max_y = int(max(c[1] + c[2] for c in row_sorted))
                
                pad_x = 5
                pad_y = 5
                x1 = max(0, min_x - pad_x)
                x2 = min(gray.shape[1], max_x + pad_x)
                y1 = max(0, min_y - pad_y)
                y2 = min(gray.shape[0], max_y + pad_y)
                
                crop = gray[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    answers[q_num] = None
                    q_num += 1
                    continue
                
                r_avg = int(np.mean([c[2] for c in row_sorted]))

                # --- 1. Xóa line ngang dọc ---
                th_inv = cv2.adaptiveThreshold(
                    crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 21, 10
                )
                kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, r_avg * 2)))
                v_lines = cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, kernel_v)
                kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, r_avg * 2), 1))
                h_lines = cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, kernel_h)
                line_mask = cv2.bitwise_or(v_lines, h_lines)
                crop[line_mask > 0] = 255
                
                # --- 2. Xóa trắng 2 bên mép (nằm ngoài A và D) ---
                crop_cx_A = int(row_sorted[0][0] - x1)
                crop_cx_D = int(row_sorted[-1][0] - x1)
                margin = int(r_avg * 1.2)
                left_bound = max(0, crop_cx_A - margin)
                crop[:, :left_bound] = 255
                right_bound = min(crop.shape[1], crop_cx_D + margin)
                crop[:, right_bound:] = 255
                
                # --- 3. Threshold lại ảnh đã clean ---
                _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # --- 4. Tính điểm mật độ pixel trên từng bubble ---
                counts = []
                for ci in range(n):
                    cxc = int(row_sorted[ci][0] - x1)
                    cyc = int(row_sorted[ci][1] - y1)
                    r = int(row_sorted[ci][2])
                    
                    mask_inner = np.zeros(binary.shape, dtype="uint8")
                    inner_r = max(int(r * 0.55), 3)
                    cv2.circle(mask_inner, (cxc, cyc), inner_r, 255, -1)
                    
                    pixel_count = cv2.countNonZero(cv2.bitwise_and(binary, binary, mask=mask_inner))
                    counts.append(pixel_count)
                    
                max_idx = np.argmax(counts)
                max_val = counts[max_idx]
                sorted_counts = sorted(counts, reverse=True)
                gap = sorted_counts[0] - sorted_counts[1] if n > 1 else max_val
                
                area = np.pi * (max(int(r_avg * 0.55), 3) ** 2)
                
                # Ngưỡng: Cần ít nhất 20% pixel đen trong khuôn và cách đối thủ ít nhất 10%
                if max_val > (area * 0.20) and gap > (area * 0.10):
                    answer = CHOICE_LABELS[max_idx]
                    answers[q_num] = answer
                else:
                    answer = None
                    answers[q_num] = None
                    
                # Visualize on clean crop
                vis_crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                for ci in range(n):
                    cxc = int(row_sorted[ci][0] - x1)
                    cyc = int(row_sorted[ci][1] - y1)
                    r = int(row_sorted[ci][2])
                    color = (0, 0, 255) if (answer and ci == CHOICE_LABELS.index(answer)) else (0, 200, 0)
                    cv2.circle(vis_crop, (cxc, cyc), max(int(r * 0.55), 3), color, 2)
                    
                question_images[q_num] = {
                    'image': vis_crop,
                    'bbox': (x1, y1, x2, y2),
                    'answer': answers[q_num],
                    'circles': [(c[0]-x1, c[1]-y1, c[2]) for c in row_sorted]
                }
                
                # Vẽ lên vis_thresh
                for ci, (cx, cy, r) in enumerate(row[:NUM_CHOICES]):
                    if answer and ci < len(CHOICE_LABELS) and CHOICE_LABELS[ci] == answer:
                        cv2.circle(vis_thresh, (cx, cy), r + 5, (0, 0, 255), 3)
                        cv2.circle(vis_thresh, (cx, cy), r, (0, 0, 255), -1)
                    else:
                        cv2.circle(vis_thresh, (cx, cy), r, (0, 200, 0), 1)
                
                label = f"{q_num}:{answer or '-'}"
                cv2.putText(vis_thresh, label,
                            (row[0][0] - row[0][2] - 70, row[0][1] + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
                q_num += 1

        n_answered = sum(1 for v in answers.values() if v is not None)
        print(f"  -> {n_answered}/{len(answers)} câu có đáp án")

        self._add_step(
            "Threshold Grading (Crop & Clean)",
            f"Điểm mật độ pixel sau morphology line removal.\n"
            f"Kết quả: {n_answered}/{len(answers)} câu có đáp án",
            vis_thresh)

        # === BƯỚC 6: Bỏ qua vì đã sinh question_images ở trên ===

        annotated = self._draw_annotated(image, grid, answers)
        self._add_step(
            "Kết quả Annotated",
            f"Ảnh kết quả cuối cùng với vòng tròn đáp án\n"
            f"và số thứ tự câu hỏi.",
            annotated)

        elapsed = time.time() - t0

        stats = {
            'total_circles': len(raw_circles),
            'filtered_circles': len(good_circles),
            'total_questions': NUM_QUESTIONS,
            'detected_questions': total_q,
            'answered': n_answered,
            'unanswered': total_q - n_answered,
            'processing_time': elapsed
        }

        self._print_results(answers, stats)
        self._gray_clahe_cache = None

        return {
            'answers': answers,
            'annotated': annotated,
            'question_images': question_images,
            'stats': stats,
            'grid': grid,
            'image_orig': image,
            'step_images': list(self._step_images)
        }


    def _detect_circles(self, gray):
        """Phát hiện toàn bộ vòng tròn bằng HoughCircles (2 lượt)."""
        # Dùng CLAHE đã cache để giảm chi phí tính toán.
        enhanced = self._gray_clahe_cache if self._gray_clahe_cache is not None else gray
        blurred = cv2.GaussianBlur(enhanced, (9, 9), 2)
        
        # Lượt detect chính theo bộ tham số chuẩn.
        result = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
            param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
            minRadius=HOUGH_MIN_RADIUS, maxRadius=HOUGH_MAX_RADIUS
        )
        circles = []
        if result is not None:
            circles = [(int(x), int(y), int(r)) 
                       for x, y, r in np.round(result[0]).astype(int)]
        
        # Lượt bổ sung với ngưỡng nới lỏng nếu detect thiếu.
        if len(circles) < NUM_QUESTIONS * NUM_CHOICES * 0.6:
            result2 = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT,
                dp=1.2, minDist=25,
                param1=40, param2=25,
                minRadius=HOUGH_MIN_RADIUS - 3, maxRadius=HOUGH_MAX_RADIUS + 5
            )
            if result2 is not None:
                existing = set((x, y) for x, y, r in circles)
                for x, y, r in np.round(result2[0]).astype(int):
                    x, y, r = int(x), int(y), int(r)
                    # Chỉ thêm circle mới nếu chưa trùng vị trí circle cũ.
                    if not any(abs(x-ex) < 15 and abs(y-ey) < 15 
                              for ex, ey in existing):
                        circles.append((x, y, r))
                        existing.add((x, y))
        return circles

    # ===================================================================
    # BƯỚC 2: LỌC THEO BÁN KÍNH + ĐỘ TRÒN
    # ===================================================================
    def _filter_by_radius(self, circles, gray=None):
        """Lọc circle outlier bằng median radius và kiểm tra độ tròn cục bộ."""
        if not circles:
            return []
        radii = [c[2] for c in circles]
        med = np.median(radii)
        filtered = [(x, y, r) for x, y, r in circles if abs(r - med) < med * 0.35]
        
        # Lọc bổ sung để loại marker vuông/nhiễu bằng circularity.
        if gray is not None and len(filtered) > 20:
            validated = []
            h, w = gray.shape
            for x, y, r in filtered:
                # Cắt ROI quanh circle để kiểm tra cục bộ.
                x1, y1 = max(0, x - r - 2), max(0, y - r - 2)
                x2, y2 = min(w, x + r + 3), min(h, y + r + 3)
                crop = gray[y1:y2, x1:x2]
                if crop.size < 10:
                    continue
                
                # Đánh giá circularity qua contour trong ROI.
                _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(cnt)
                    perim = cv2.arcLength(cnt, True)
                    if perim > 0:
                        circularity = 4 * np.pi * area / (perim * perim)
                        # Circle thật thường có circularity cao hơn nhiễu không đều.
                        if circularity < 0.35:
                            continue
                
                validated.append((x, y, r))
            
            if len(validated) >= len(filtered) * 0.6:  # tránh lọc quá gắt
                return validated
        
        return filtered

    # ===================================================================
    # BƯỚC 3: CHIA 4 CỘT CHÍNH
    # ===================================================================
    def _split_into_main_columns(self, circles):
        """
        Chia circles thành 4 cột chính.
        Bước 1: Cluster X → subcols (nhóm circles cùng vị trí X)
        Bước 2: Lọc subcols hợp lệ (20-40 circles = answer subcols)  
        Bước 3: Tìm 3 gaps lớn nhất giữa subcol centers → 4 cột
        Bước 4: Gán circles theo column boundaries
        """
        if not circles:
            return [[] for _ in range(4)]

        # Bước 1: Cluster X thành subcols
        sorted_c = sorted(circles, key=lambda c: c[0])
        subcols = []
        current = [sorted_c[0]]
        for i in range(1, len(sorted_c)):
            if sorted_c[i][0] - np.mean([c[0] for c in current]) > 35:
                subcols.append(current)
                current = [sorted_c[i]]
            else:
                current.append(sorted_c[i])
        subcols.append(current)

        # Bước 2: Lọc subcols có 20-40 circles (likely answer subcols)
        valid = [(np.mean([c[0] for c in sc]), sc) for sc in subcols if 20 <= len(sc) <= 40]
        valid.sort(key=lambda x: x[0])

        if len(valid) < 4:
            # Fallback: hạ ngưỡng
            valid = [(np.mean([c[0] for c in sc]), sc) for sc in subcols if len(sc) >= 10]
            valid.sort(key=lambda x: x[0])

        if self.debug:
            print(f"    {len(subcols)} subcols, {len(valid)} valid (20-40 circles)")

        # Bước 3: Tìm 3 gaps lớn nhất giữa valid subcol centers
        centers = [v[0] for v in valid]
        gaps = [(centers[i+1] - centers[i], i) for i in range(len(centers)-1)]
        gaps.sort(reverse=True)
        split_indices = sorted([g[1] for g in gaps[:3]])

        # Determine column X boundaries (midpoints of gaps)
        boundaries = []
        for si in split_indices:
            mid = (centers[si] + centers[si+1]) / 2
            boundaries.append(mid)

        if self.debug:
            print(f"    Column boundaries: {[f'{b:.0f}' for b in boundaries]}")

        # Bước 4: Gán TẤT CẢ circles (không chỉ valid subcols) theo boundaries
        columns = [[] for _ in range(4)]
        for c in circles:
            col_idx = 0
            for b in boundaries:
                if c[0] > b:
                    col_idx += 1
            if col_idx < 4:
                columns[col_idx].append(c)

        return columns

    # ===================================================================
    # BƯỚC 4: Xây dựng grid đáp án
    # ===================================================================
    def _build_answer_grid(self, columns, gray):
        """
        Cho mỗi cột chính, phân thành hàng (mỗi hàng = 1 câu = 4 bubbles ABCD).
        Bỏ hàng header (ABCD label in sẵn ở đầu cột).
        
        Trả về: {col_idx: [row0, row1, ...]}
            Mỗi row = [(cx,cy,r), ...] đã sort trái→phải (chỉ 4 circle ABCD)
        """
        grid = {}

        # Trước tiên, xác định spacing tham chiếu từ tất cả cột.
        ref_subcol_spacings = []
        ref_y_starts = []
        ref_y_ends = []
        
        for col_idx in range(len(columns)):
            col_circles = columns[col_idx]
            if len(col_circles) < 20:
                continue
            subcols = self._cluster_x_local(col_circles, threshold=30)
            # Gộp các subcol quá gần nhau.
            if len(subcols) > 1:
                merged = [list(subcols[0])]
                for i in range(1, len(subcols)):
                    prev_center = np.mean([c[0] for c in merged[-1]])
                    curr_center = np.mean([c[0] for c in subcols[i]])
                    if curr_center - prev_center < 25:
                        merged[-1].extend(subcols[i])
                    else:
                        merged.append(list(subcols[i]))
                subcols = merged
            valid_sc = [sc for sc in subcols if len(sc) >= 20]
            if len(valid_sc) >= 4:
                # Sắp theo tâm X, lấy bộ 4 đại diện.
                sc_sorted = sorted(valid_sc, key=lambda sc: np.mean([c[0] for c in sc]))[:4]
                centers = [np.mean([c[0] for c in sc]) for sc in sc_sorted]
                spacings = [centers[i+1] - centers[i] for i in range(3)]
                # Chỉ dùng khi spacing tương đối đều.
                if min(spacings) > 0 and max(spacings) / min(spacings) < 1.5:
                    ref_subcol_spacings.extend(spacings)
        
        avg_subcol_spacing = np.median(ref_subcol_spacings) if ref_subcol_spacings else 115

        # Bước 1: Xử lý cột 1-3 trước, thu thập Y range hợp lệ
        for col_idx in range(min(3, len(columns))):
            col_circles = columns[col_idx]
            if not col_circles or len(col_circles) < 10:
                grid[col_idx] = []
                continue
            rows = self._process_one_column(col_idx, col_circles, gray, avg_subcol_spacing)
            grid[col_idx] = rows
            if rows:
                all_ys = [c[1] for row in rows for c in row]
                ref_y_starts.append(min(all_ys))
                ref_y_ends.append(max(all_ys))

        # Xác định Y range cho answer area từ cột 1-3
        if ref_y_starts and ref_y_ends:
            y_min_answer = min(ref_y_starts) - 50   # cho phép lệch nhỏ
            y_max_answer = max(ref_y_ends) + 50
        else:
            h = gray.shape[0]
            y_min_answer = int(h * 0.2)
            y_max_answer = int(h * 0.95)

        if self.debug:
            print(f"    Khoang Y dap an: [{y_min_answer:.0f}, {y_max_answer:.0f}]")

        # Tính độ rộng ABCD tham chiếu từ cột 1-3 để sửa biên cột 4.
        ref_abcd_d_positions = []  # Tâm X subcol D của cột 1-3
        ref_abcd_widths = []
        for col_idx in range(min(3, len(columns))):
            rows_c = grid.get(col_idx, [])
            if not rows_c:
                continue
            # Lấy tâm subcol từ các hàng có đủ 4 circle.
            four_circle_rows = [r for r in rows_c if len(r) >= 4]
            if len(four_circle_rows) >= 10:
                a_xs = [r[0][0] for r in four_circle_rows]
                d_xs = [r[3][0] for r in four_circle_rows]
                ref_abcd_d_positions.append(np.median(d_xs))
                ref_abcd_widths.append(np.median(d_xs) - np.median(a_xs))

        expected_abcd_width = np.median(ref_abcd_widths) if ref_abcd_widths else avg_subcol_spacing * 3

        # Bước 2: Xử lý cột 4 với Y filter + boundary correction
        for col_idx in range(3, len(columns)):
            col_circles = columns[col_idx]
            if not col_circles or len(col_circles) < 10:
                grid[col_idx] = []
                continue
            
            # Lọc circles theo Y range (bỏ student ID/header ở top)
            filtered = [(x, y, r) for x, y, r in col_circles 
                       if y_min_answer <= y <= y_max_answer]
            
            # Sửa biên: kiểm tra cột 4 có thiếu subcol A hay không.
            if col_idx == 3 and ref_abcd_widths and col_idx - 1 in grid:
                col4_subcols = self._cluster_x_local(filtered, threshold=20) if filtered else []
                col4_valid = sorted([sc for sc in col4_subcols if len(sc) >= 15],
                                   key=lambda sc: np.mean([c[0] for c in sc]))
                if len(col4_valid) >= 3:
                    col4_width = np.mean([c[0] for c in col4_valid[-1]]) - np.mean([c[0] for c in col4_valid[0]])
                    if col4_width < expected_abcd_width * 0.85:
                        # Cột 4 quá hẹp -> khả năng mất subcol trái nhất.
                        # Thử chuyển circle phù hợp từ cột 3 sang cột 4.
                        col3_circles = columns[col_idx - 1]
                        col4_first_x = np.mean([c[0] for c in col4_valid[0]])
                        expected_a_x = col4_first_x - avg_subcol_spacing
                        # Chuyển các circle gần vị trí A kỳ vọng.
                        stolen = []
                        remaining_col3 = []
                        for c in col3_circles:
                            if abs(c[0] - expected_a_x) < avg_subcol_spacing * 0.4:
                                stolen.append(c)
                            else:
                                remaining_col3.append(c)
                        if len(stolen) >= 10:
                            # Chỉ giữ circle trong dải Y đáp án.
                            stolen_y = [(x, y, r) for x, y, r in stolen
                                       if y_min_answer <= y <= y_max_answer]
                            filtered = stolen_y + filtered
                            columns[col_idx - 1] = remaining_col3
                            if self.debug:
                                print(f"    Boundary fix: moved {len(stolen_y)} circles from Col3 to Col4")
                            # Chạy lại cột 3 sau khi đã điều chỉnh biên.
                            grid[col_idx - 1] = self._process_one_column(
                                col_idx - 1, remaining_col3, gray, avg_subcol_spacing)
            
            if self.debug:
                print(f"    Cột {col_idx+1}: {len(col_circles)} → {len(filtered)} circles (Y filter)")
            
            rows = self._process_one_column(col_idx, filtered, gray, avg_subcol_spacing)
            grid[col_idx] = rows

        return grid

    def _process_one_column(self, col_idx, col_circles, gray, avg_subcol_spacing):
        """Xử lý 1 cột: tách ABCD → gom hàng theo Y → bỏ header → kiểm tra."""
        valid_rows = self._process_one_column_inner(col_idx, col_circles, gray, avg_subcol_spacing)
        
        # Nếu thiếu hàng, thử lại với ngưỡng subcol nới lỏng để chịu méo phối cảnh tốt hơn.
        if len(valid_rows) < QUESTIONS_PER_COLUMN * 0.85:
            retry = self._process_one_column_inner(col_idx, col_circles, gray, avg_subcol_spacing, subcol_threshold=30)
            if len(retry) > len(valid_rows):
                if self.debug:
                    print(f"    Cột {col_idx+1}: fallback {len(valid_rows)}→{len(retry)} hàng (nới threshold)")
                valid_rows = retry
        
        return valid_rows

    def _process_one_column_inner(self, col_idx, col_circles, gray, avg_subcol_spacing, subcol_threshold=20):
        """Xử lý lõi cho 1 cột với ngưỡng tách subcol có thể điều chỉnh."""
        # Tìm 4 subcol ABCD
        abcd_circles = self._extract_abcd_circles(col_circles, avg_subcol_spacing, threshold=subcol_threshold)

        # Cluster Y → hàng
        rows_raw = self._cluster_y(abcd_circles)
        
        if self.debug:
            print(f"    Cột {col_idx+1}: {len(rows_raw)} hàng thô ({len(abcd_circles)} circles)")

        # Bỏ header row(s)
        rows_clean = self._remove_header_rows(rows_raw, gray)
        
        # Chỉ giữ hàng có 3-6 circles  
        valid_rows = []
        for row in rows_clean:
            row_sorted = sorted(row, key=lambda c: c[0])
            if 3 <= len(row_sorted) <= 6:
                if len(row_sorted) > 4:
                    row_sorted = self._pick_best_4_from_n(row_sorted)
                valid_rows.append(row_sorted)

        # Nếu >30 hàng thì bỏ phần dư từ đầu (thường là header lọt vào).
        if len(valid_rows) > QUESTIONS_PER_COLUMN:
            excess = len(valid_rows) - QUESTIONS_PER_COLUMN
            if excess <= 2:  # Chỉ 1-2 hàng thừa: header chưa lọc được
                valid_rows = valid_rows[excess:]
                if self.debug:
                    print(f"    Cột {col_idx+1}: bỏ {excess} hàng đầu (header fallback)")
            else:  # Nhiều hàng thừa: dùng spacing selection
                valid_rows = self._select_best_rows(valid_rows, QUESTIONS_PER_COLUMN)
        
        if self.debug:
            print(f"    Cột {col_idx+1}: {len(valid_rows)} hàng hợp lệ")

        return valid_rows

    def _cluster_x_local(self, circles, threshold=35):
        """Gom circle theo trục X trong phạm vi một cột chính."""
        sorted_c = sorted(circles, key=lambda c: c[0])
        subcols = []
        current = [sorted_c[0]]
        for i in range(1, len(sorted_c)):
            if sorted_c[i][0] - np.mean([c[0] for c in current]) > threshold:
                subcols.append(current)
                current = [sorted_c[i]]
            else:
                current.append(sorted_c[i])
        subcols.append(current)
        return subcols

    def _extract_abcd_circles(self, col_circles, avg_spacing, threshold=20):
        """
        Từ tất cả circles trong 1 cột chính, chọn ra 4 nhóm subcol ABCD.
        Chiến lược:
        1. Gom subcol theo X với ngưỡng cấu hình.
        2. Lọc subcol nhiễu (quá ít circle).
        3. Nếu >4 subcol hợp lệ thì chọn bộ 4 có spacing đều nhất.
        4. Nếu thiếu subcol thì ước lượng theo khoảng cách trung bình.
        """
        if len(col_circles) < 8:
            return col_circles
            
        # Bước chính: gom subcol theo ngưỡng threshold.
        subcols = self._cluster_x_local(col_circles, threshold=threshold)
        
        # Gộp subcol quá sát nhau (< 20% spacing trung bình) để tránh tách đôi cùng một cột.
        merge_dist = max(avg_spacing * 0.2, 15)
        if len(subcols) > 1:
            merged = [list(subcols[0])]
            for i in range(1, len(subcols)):
                prev_center = np.mean([c[0] for c in merged[-1]])
                curr_center = np.mean([c[0] for c in subcols[i]])
                if curr_center - prev_center < merge_dist:
                    merged[-1].extend(subcols[i])
                else:
                    merged.append(list(subcols[i]))
            subcols = merged

        # Giữ subcol có đủ circle để đại diện cột bubble thật.
        valid_subcols = [sc for sc in subcols if len(sc) >= 15]
        
        if len(valid_subcols) == 4:
            result = []
            for sc in valid_subcols:
                result.extend(sc)
            return result
        
        if len(valid_subcols) > 4:
            # Chọn 4 subcol có spacing đều nhất.
            sc_data = [(np.mean([c[0] for c in sc]), len(sc), sc) for sc in valid_subcols]
            sc_data.sort(key=lambda x: x[0])
            
            from itertools import combinations
            best_score = -1
            best_group = None
            for combo in combinations(range(len(sc_data)), 4):
                centers = [sc_data[i][0] for i in combo]
                counts = [sc_data[i][1] for i in combo]
                total_circles = sum(counts)
                spacings = [centers[i+1] - centers[i] for i in range(3)]
                spacing_var = np.var(spacings) if spacings else 0
                avg_sp = np.mean(spacings) if spacings else 0
                spacing_penalty = abs(avg_sp - avg_spacing) / avg_spacing if avg_spacing > 0 else 0
                score = total_circles / (1 + spacing_var / 1000 + spacing_penalty * 2)
                if score > best_score:
                    best_score = score
                    best_group = combo
            
            if best_group:
                result = []
                for i in best_group:
                    result.extend(sc_data[i][2])
                return result
        
        if len(valid_subcols) == 3:
            # Thiếu 1 subcol: dùng spacing từ 3 subcol hiện có để suy ra vị trí còn thiếu.
            centers3 = sorted([np.mean([c[0] for c in sc]) for sc in valid_subcols])
            spacings = [centers3[i+1] - centers3[i] for i in range(len(centers3)-1)]
            sp = np.median(spacings) if spacings else avg_spacing
            
            # Xác định vị trí thiếu ở giữa/đầu/cuối dựa trên cấu trúc khoảng cách.
            all_xs = [c[0] for c in col_circles]
            x_min, x_max = min(all_xs), max(all_xs)
            
            # Kiểm tra khoảng trống lớn bất thường ở giữa.
            found_double = False
            for i in range(len(spacings)):
                if spacings[i] > sp * 1.5:
                    # Có gap lớn ở giữa 2 tâm -> chèn tâm mới vào giữa.
                    mid = (centers3[i] + centers3[i+1]) / 2
                    # Tạo tâm subcol thứ 4 (ước lượng).
                    centers4 = sorted(centers3 + [mid])
                    found_double = True
                    break
            
            if not found_double:
                # Nếu không thiếu ở giữa thì xét khả năng thiếu ở mép trái/phải.
                left_gap = centers3[0] - x_min
                right_gap = x_max - centers3[-1]
                if left_gap > sp * 0.5:
                    centers4 = sorted([centers3[0] - sp] + centers3)
                elif right_gap > sp * 0.5:
                    centers4 = sorted(centers3 + [centers3[-1] + sp])
                else:
                    centers4 = centers3
            
            # Gán circle về tâm gần nhất trong 4 tâm ước lượng.
            if len(centers4) == 4:
                half = sp * 0.6
                result = []
                for c in col_circles:
                    dists = [abs(c[0] - ctr) for ctr in centers4]
                    if min(dists) < half:
                        result.append(c)
                return result
        
        # Fallback cuối: trả toàn bộ circle của cột để không mất dữ liệu.
        return col_circles

    def _cluster_y(self, circles):
        """Gom circles thành hàng dựa trên Y."""
        sorted_c = sorted(circles, key=lambda c: c[1])
        
        ys = [c[1] for c in sorted_c]
        diffs = [ys[i+1] - ys[i] for i in range(len(ys)-1)]
        if not diffs:
            return [sorted_c]

        # Tìm ngưỡng tách hàng dựa trên "khe" đầu tiên đủ lớn giữa các Y-diff.
        # Cách này bền vững hơn chọn "gap lớn nhất" vì ít bị outlier chi phối.
        sorted_diffs = sorted(diffs)
        split_val = 30  # default
        for i in range(len(sorted_diffs) - 1):
            if sorted_diffs[i + 1] - sorted_diffs[i] > 20:
                split_val = (sorted_diffs[i] + sorted_diffs[i + 1]) / 2
                break

        y_threshold = max(split_val, 15)

        rows = []
        current = [sorted_c[0]]
        for i in range(1, len(sorted_c)):
            if sorted_c[i][1] - current[-1][1] < y_threshold:
                current.append(sorted_c[i])
            else:
                rows.append(current)
                current = [sorted_c[i]]
        rows.append(current)
        return rows

    def _remove_header_rows(self, rows, gray):
        """
        Bỏ hàng header ở đầu cột (label A B C D in sẵn trên form).
        Sử dụng CLAHE để chuẩn hóa độ sáng trước khi kiểm tra.
        Dùng ROI crop nhỏ thay vì full-image mask để tăng tốc.
        """
        if not rows:
            return rows

        # Dùng CLAHE đã cache (hoặc tạo mới nếu chưa có)
        if not hasattr(self, '_gray_clahe_cache') or self._gray_clahe_cache is None:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            self._gray_clahe_cache = clahe.apply(gray)
        gray_norm = self._gray_clahe_cache

        start_idx = 0
        max_header_check = 2

        while start_idx < min(max_header_check, len(rows)):
            row = rows[start_idx]

            if len(row) < 3:
                start_idx += 1
                continue

            # Tính mean grayscale bằng ROI crop nhỏ (tối ưu tốc độ)
            means = []
            h, w = gray_norm.shape
            for cx, cy, r in row:
                cx, cy, r = int(cx), int(cy), int(r)
                ir = max(int(r * 0.4), 3)
                x1, y1 = max(0, cx - ir), max(0, cy - ir)
                x2, y2 = min(w, cx + ir + 1), min(h, cy + ir + 1)
                roi = gray_norm[y1:y2, x1:x2]
                if roi.size < 4:
                    means.append(200.0)
                    continue
                rh, rw = roi.shape
                mask = np.zeros((rh, rw), dtype="uint8")
                cv2.circle(mask, (cx - x1, cy - y1), ir, 255, -1)
                mean_val = cv2.mean(roi, mask=mask)[0]
                means.append(mean_val)

            all_dark = all(m < FILLED_MEAN_THRESHOLD for m in means)
            any_light = any(m > EMPTY_MEAN_THRESHOLD for m in means)
            brightness_range = max(means) - min(means)

            cond1 = all_dark and not any_light

            has_pencil_fill = any(m < 130 for m in means)
            all_dim        = all(m < 195 for m in means)
            low_contrast   = brightness_range < 45
            cond2 = low_contrast and all_dim and not has_pencil_fill

            if cond1 or cond2:
                if self.debug:
                    ms = ', '.join(f'{m:.0f}' for m in means)
                    cname = 'cond1' if cond1 else 'cond2'
                    print(f"    [Skip header/{cname}] y~{row[0][1]}, means=[{ms}], range={brightness_range:.0f}")
                start_idx += 1
            else:
                break

        return rows[start_idx:]

    def _pick_best_4_from_n(self, sorted_circles):
        """Từ N circles (>4), chọn 4 cái có khoảng cách X đều nhau nhất."""
        from itertools import combinations
        if len(sorted_circles) <= 4:
            return sorted_circles

        best_4 = None
        best_score = float('inf')
        for combo in combinations(range(len(sorted_circles)), 4):
            sub = [sorted_circles[j] for j in combo]
            sub_sorted = sorted(sub, key=lambda c: c[0])
            gaps = [sub_sorted[i+1][0] - sub_sorted[i][0] for i in range(3)]
            score = np.var(gaps)
            if score < best_score:
                best_score = score
                best_4 = sub_sorted
        return best_4

    def _select_best_rows(self, rows, target_count):
        """Chọn target_count hàng đều nhất."""
        if len(rows) <= target_count:
            return rows

        row_ys = [(np.mean([c[1] for c in row]), i) for i, row in enumerate(rows)]
        row_ys.sort()

        total_span = row_ys[-1][0] - row_ys[0][0]
        expected_spacing = total_span / (target_count - 1)

        selected_indices = []
        used = set()
        for t in range(target_count):
            target_y = row_ys[0][0] + t * expected_spacing
            best_idx = None
            best_dist = float('inf')
            for y_val, idx in row_ys:
                if idx in used:
                    continue
                dist = abs(y_val - target_y)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is not None:
                selected_indices.append(best_idx)
                used.add(best_idx)

        selected_indices.sort()
        return [rows[i] for i in selected_indices]

    # ===================================================================
    # BƯỚC 5: CHẤM ĐÁP ÁN (ROI CỤC BỘ ĐỂ TỐI ƯU TỐC ĐỘ)
    # ===================================================================
    @staticmethod
    def _bubble_contrast(gray, cx, cy, r):
        """
        Tính inner_mean và outer_mean cho 1 bubble bằng ROI crop nhỏ.
        Nhanh hơn ~100x so với tạo np.zeros(gray.shape) full-image.
        Returns:
            (inner_val, outer_val, contrast_ratio)
        """
        cx, cy, r = int(cx), int(cy), int(r)
        h, w = gray.shape
        outer_r = min(int(r * 2.0), r + 30)
        # Dùng inner radius nhỏ để lấy vùng lõi thật của bubble,
        # tránh viền in sẵn gây false-positive ở ô chưa tô.
        inner_r = max(int(r * 0.4), 3)
        outer_inner = max(int(r * 1.3), r + 3)

        # ROI nhỏ quanh bubble để tính nhanh hơn mask full-image.
        x1 = max(0, cx - outer_r)
        y1 = max(0, cy - outer_r)
        x2 = min(w, cx + outer_r + 1)
        y2 = min(h, cy + outer_r + 1)
        roi = gray[y1:y2, x1:x2]
        if roi.size < 4:
            return 200.0, 200.0, 1.0

        rh, rw = roi.shape
        lcx, lcy = cx - x1, cy - y1  # local center

        # Mask vùng lõi bubble.
        inner_mask = np.zeros((rh, rw), dtype="uint8")
        cv2.circle(inner_mask, (lcx, lcy), inner_r, 255, -1)
        inner_val = cv2.mean(roi, mask=inner_mask)[0]

        # Mask vành ngoài để so tương phản với vùng lõi.
        outer_mask = np.zeros((rh, rw), dtype="uint8")
        cv2.circle(outer_mask, (lcx, lcy), outer_r, 255, -1)
        cv2.circle(outer_mask, (lcx, lcy), outer_inner, 0, -1)
        outer_val = cv2.mean(roi, mask=outer_mask)[0]

        if outer_val < 10:
            outer_val = 200.0
        contrast = inner_val / outer_val
        return inner_val, outer_val, contrast

    @staticmethod
    def _bubble_fill_ratio(gray, cx, cy, r, threshold=80):
        """Tính tỉ lệ pixel tối (< threshold) trong vùng trung tâm bubble.
        
        Gợi ý thực nghiệm:
        - Tô thật: fr80 thường cao (nhiều pixel < threshold).
        - Nhiễu nhẹ: fr80 thấp (đa số pixel chưa đủ tối).
        """
        cx, cy, r = int(cx), int(cy), int(r)
        h, w = gray.shape
        inner_r = max(int(r * 0.4), 3)
        x1 = max(0, cx - inner_r - 1)
        y1 = max(0, cy - inner_r - 1)
        x2 = min(w, cx + inner_r + 2)
        y2 = min(h, cy + inner_r + 2)
        roi = gray[y1:y2, x1:x2]
        rh, rw = roi.shape
        lcx, lcy = cx - x1, cy - y1
        mask = np.zeros((rh, rw), dtype='uint8')
        cv2.circle(mask, (lcx, lcy), inner_r, 255, -1)
        total = cv2.countNonZero(mask)
        if total == 0:
            return 0.0
        pixels = roi[mask > 0]
        return float(np.sum(pixels < threshold)) / total

    def _grade_all(self, gray, grid):
        """Chấm toàn bộ câu bằng phân tích tương đối theo từng hàng.
        
        Nguyên tắc chính:
        1. Trong mỗi hàng ABCD, tìm bubble tối nhất.
        2. Bubble được coi là tô khi tối hơn rõ rệt so với phương án kế tiếp.
        3. Có cơ chế recover khi hàng thiếu 1 circle.
        """
        answers = {}
        q_num = 1
        for col_idx in sorted(grid.keys()):
            col_rows = grid[col_idx]
            
            # Ước lượng tâm X chuẩn của 4 subcol từ các hàng đủ 4 circle.
            subcol_xs = [[] for _ in range(NUM_CHOICES)]
            median_r = []
            for row in col_rows:
                if len(row) >= 4:
                    row_sorted = sorted(row[:4], key=lambda c: c[0])
                    for ci in range(4):
                        subcol_xs[ci].append(row_sorted[ci][0])
                        median_r.append(row_sorted[ci][2])
            expected_xs = [np.median(xs) if xs else None for xs in subcol_xs]
            avg_r = np.median(median_r) if median_r else 20
            
            for ri, row in enumerate(col_rows):
                if q_num > NUM_QUESTIONS:
                    break
                n = min(len(row), NUM_CHOICES)
                if n < 3:
                    answers[q_num] = None
                    q_num += 1
                    continue
                
                inner_vals = []
                contrasts = []
                for ci in range(n):
                    cx, cy, r = row[ci]
                    iv, ov, cr = self._bubble_contrast(gray, cx, cy, r)
                    inner_vals.append(iv)
                    contrasts.append(cr)
                
                answer = self._grade_row(inner_vals, contrasts, gray, row)
                
                # Recovery: hàng có 3 circle và chưa có đáp án -> thử vị trí subcol bị thiếu.
                if answer is None and n == 3 and all(x is not None for x in expected_xs):
                    row_xs = [row[ci][0] for ci in range(3)]
                    row_y = int(np.mean([row[ci][1] for ci in range(3)]))
                    r = int(avg_r)
                    
                    # Xác định subcol nào đang bị thiếu.
                    matched = [False] * 4
                    for rx in row_xs:
                        best_ci = min(range(4), key=lambda ci: abs(rx - expected_xs[ci]))
                        matched[best_ci] = True
                    missing = [ci for ci in range(4) if not matched[ci]]
                    
                    if len(missing) == 1:
                        mi = missing[0]
                        base_mx = int(expected_xs[mi])
                        best_cr, best_iv, best_dx = 1.0, 255, 0
                        for dx in range(-15, 16, 5):
                            mx = base_mx + dx
                            iv, ov, cr = self._bubble_contrast(gray, mx, row_y, r)
                            if cr < best_cr:
                                best_cr, best_iv, best_dx = cr, iv, dx
                        if best_cr < 0.72 and best_iv < 180:
                            answer = CHOICE_LABELS[mi]
                            if self.debug:
                                print(f"    Q{q_num:3d}: phuc hoi {answer} tai subcol thieu (iv={best_iv:.0f}, cr={best_cr:.2f}, dx={best_dx})")
                
                answers[q_num] = answer
                
                if self.debug:
                    a = answer if answer else "-"
                    ivs = ', '.join(f'{v:.0f}' for v in inner_vals)
                    cs = ', '.join(f'{c:.2f}' for c in contrasts)
                    print(f"    Q{q_num:3d}: {a}  inner=[{ivs}] cr=[{cs}]")
                q_num += 1
        return answers

    def _grade_row(self, inner_vals, contrasts, gray=None, row=None):
        """Chấm 1 hàng bằng phân tích độ tối tương đối giữa các lựa chọn."""
        n = min(len(inner_vals), NUM_CHOICES)
        if n < 3:
            return None
        
        # Tìm bubble tối nhất trong hàng.
        vals = inner_vals[:n]
        sorted_vals = sorted(vals)
        min_val = sorted_vals[0]
        second_val = sorted_vals[1]
        min_idx = vals.index(min_val)
        
        # Nếu bubble tối nhất vẫn sáng -> xem như chưa tô.
        if min_val > 185:
            return None
        
        # Khoảng cách độ tối giữa bubble tối nhất và bubble thứ 2.
        gap = second_val - min_val
        
        # Điều kiện 1: phải tối hơn rõ rệt so với phương án kế cận.
        if gap < 40:
            return None
        
        # Điều kiện 2: vùng ranh giới thì kiểm tra thêm fill ratio để loại nhiễu.
        if min_val >= 90 and gray is not None and row is not None and min_idx < len(row):
            cx, cy, r = row[min_idx]
            fr = self._bubble_fill_ratio(gray, cx, cy, r, threshold=80)
            if fr < 0.20:
                return None
        
        # Điều kiện 3: tỷ lệ tương phản phải đủ thấp.
        # Mức 1: bằng chứng mạnh.
        if contrasts[min_idx] < 0.72:
            return CHOICE_LABELS[min_idx]
        
        # Mức 2: tô nhạt, yêu cầu thêm ràng buộc để tránh false-positive.
        if contrasts[min_idx] < 0.82 and min_val < 160 and gap >= 50:
            return CHOICE_LABELS[min_idx]
        
        return None

    # ===================================================================
    # PHÁT HIỆN NHIỄU THEO CỘT (CONTAMINATION)
    # ===================================================================
    def _detect_and_fix_contamination(self, answers, gray, grid):
        """
        Phát hiện subcol bị nhiễu (contamination) do text số câu hoặc viền cột.
        
        Nếu 1 đáp án thống trị >80% hàng trong 1 cột chính VÀ các subcol khác
        không bao giờ có dấu hiệu được tô thật → subcol đó bị nhiễu.
        
        Khi phát hiện contamination:
        - Nếu inner value của subcol nhiễu THẤP HƠN NHIỀU so với mức nhiễu trung bình
          → đó là fill thật (giữ nguyên)
        - Ngược lại → set blank
        """
        for col_idx in sorted(grid.keys()):
            start_q = col_idx * QUESTIONS_PER_COLUMN + 1
            end_q = min(start_q + QUESTIONS_PER_COLUMN, NUM_QUESTIONS + 1)
            
            # Đếm phân bố đáp án
            answer_counts = {}
            for q in range(start_q, end_q):
                a = answers.get(q)
                if a is not None:
                    answer_counts[a] = answer_counts.get(a, 0) + 1
            
            if not answer_counts:
                continue
            
            total_answered = sum(answer_counts.values())
            if total_answered == 0:
                continue
            
            dominant_ans = max(answer_counts, key=answer_counts.get)
            dominant_count = answer_counts[dominant_ans]
            
            # Cần thống trị mạnh: >80% cùng đáp án
            if dominant_count < total_answered * 0.8:
                continue
            
            ci_dominant = CHOICE_LABELS.index(dominant_ans)
            rows = grid[col_idx]
            
            # Thu thập inner values của subcol nhiễu và kiểm tra subcol khác
            dominant_inner_vals = []
            other_dark_count = 0
            for row in rows:
                n = min(len(row), NUM_CHOICES)
                if ci_dominant < n:
                    cx, cy, r = row[ci_dominant]
                    iv, _, _ = self._bubble_contrast(gray, cx, cy, r)
                    dominant_inner_vals.append(iv)
                
                for ci in range(n):
                    if ci == ci_dominant:
                        continue
                    cx, cy, r = row[ci]
                    iv, _, _ = self._bubble_contrast(gray, cx, cy, r)
                    if iv < 120:
                        other_dark_count += 1
                        break
            
            # Cho phép 1-2 rows nhiễu, cần >= 3 rows có fill thật ở subcol khác
            if other_dark_count >= 3:
                continue
            
            # Contamination confirmed
            # Tính baseline: median inner value của subcol nhiễu
            # Real fills sẽ THẤP HƠN NHIỀU so với baseline
            dominant_inner_vals.sort()
            median_val = dominant_inner_vals[len(dominant_inner_vals) // 2]
            # Ngưỡng: nếu inner < median * 0.55 → fill thật (vd: median=70 → threshold=38)
            real_fill_threshold = median_val * 0.55
            
            fixed = 0
            kept = 0
            row_idx = 0
            for q in range(start_q, end_q):
                if answers.get(q) == dominant_ans:
                    # Kiểm tra inner value của row này
                    if row_idx < len(rows) and ci_dominant < len(rows[row_idx]):
                        cx, cy, r = rows[row_idx][ci_dominant]
                        iv, _, _ = self._bubble_contrast(gray, cx, cy, r)
                        if iv < real_fill_threshold:
                            kept += 1  # Giữ lại — fill thật
                        else:
                            answers[q] = None
                            fixed += 1
                    else:
                        answers[q] = None
                        fixed += 1
                row_idx += 1
            
            if fixed > 0:
                msg = f"  [Contamination] Cột {col_idx+1}: subcol {dominant_ans} bị nhiễu"
                msg += f" (baseline={median_val:.0f}), đã sửa {fixed} câu → blank"
                if kept > 0:
                    msg += f", giữ {kept} câu (fill thật)"
                print(msg)
        
        return answers



    # ===================================================================
    # BƯỚC 6: Cắt ảnh từng câu
    # ===================================================================
    def _crop_all_questions(self, image, gray, grid, answers):
        """Cắt ảnh từng câu theo grid đã nhận diện để phục vụ lưu file/ML."""
        h, w = image.shape[:2]
        question_images = {}
        q_num = 1
        for col_idx in sorted(grid.keys()):
            for row in grid[col_idx]:
                if q_num > NUM_QUESTIONS:
                    break
                if not row:
                    q_num += 1
                    continue

                min_x = min(c[0] - c[2] for c in row)
                max_x = max(c[0] + c[2] for c in row)
                min_y = min(c[1] - c[2] for c in row)
                max_y = max(c[1] + c[2] for c in row)

                # Thêm vùng số thứ tự bên trái để crop đủ ngữ cảnh câu hỏi.
                bubble_width = max_x - min_x
                label_extra = int(bubble_width * 0.40)

                x1 = max(0, min_x - label_extra - CROP_PAD_X)
                x2 = min(w, max_x + CROP_PAD_X)
                y1 = max(0, min_y - CROP_PAD_Y)
                y2 = min(h, max_y + CROP_PAD_Y)

                crop = image[y1:y2, x1:x2].copy()

                # Vẽ vòng đỏ lên phương án được chọn ngay trên ảnh crop.
                answer = answers.get(q_num)
                if answer and answer in CHOICE_LABELS:
                    idx = CHOICE_LABELS.index(answer)
                    if idx < len(row):
                        cx_rel = row[idx][0] - x1
                        cy_rel = row[idx][1] - y1
                        r = row[idx][2]
                        cv2.circle(crop, (cx_rel, cy_rel), r + 5, (0, 0, 255), 3)

                question_images[q_num] = {
                    'image': crop,
                    'bbox': (x1, y1, x2, y2),
                    'answer': answer,
                    'circles': [(c[0]-x1, c[1]-y1, c[2]) for c in row]
                }
                q_num += 1
        return question_images

    # ===================================================================
    # Annotate
    # ===================================================================
    def _draw_annotated(self, image, grid, answers):
        """Vẽ ảnh annotate tổng hợp: vòng tròn đáp án + số thứ tự câu."""
        ann = image.copy()
        q_num = 1
        for col_idx in sorted(grid.keys()):
            for row in grid[col_idx]:
                if q_num > NUM_QUESTIONS:
                    break
                answer = answers.get(q_num)
                for ci, (cx, cy, r) in enumerate(row[:NUM_CHOICES]):
                    if answer and ci < len(CHOICE_LABELS) and CHOICE_LABELS[ci] == answer:
                        cv2.circle(ann, (cx, cy), r + 5, (0, 0, 255), 3)
                    else:
                        cv2.circle(ann, (cx, cy), r + 2, (0, 200, 0), 1)
                if row:
                    cv2.putText(ann, str(q_num),
                                (row[0][0] - row[0][2] - 50, row[0][1] + 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                q_num += 1
        return ann

    # ===================================================================
    # In kết quả
    # ===================================================================
    def _print_results(self, answers, stats):
        """In kết quả nhận diện và thống kê theo từng cột câu hỏi."""
        print(f"\n{'='*60}")
        print(f"  KẾT QUẢ NHẬN DIỆN")
        print(f"{'='*60}")
        for col in range(NUM_COLUMNS):
            start = col * QUESTIONS_PER_COLUMN + 1
            end = start + QUESTIONS_PER_COLUMN
            print(f"\n  Cột {col+1} (Câu {start}-{end-1}):")
            line = "  "
            for q in range(start, end):
                a = answers.get(q, "?")
                if a is None:
                    a = "-"
                line += f"{q:3d}:{a} "
                if (q - start + 1) % 10 == 0:
                    print(line)
                    line = "  "
            if line.strip():
                print(line)
        print(f"\n{'='*60}")
        print(f"  Circles: {stats['filtered_circles']}, "
              f"Câu: {stats['detected_questions']}")
        print(f"  Có đáp án: {stats['answered']}, "
              f"Bỏ trống: {stats['unanswered']}")
        print(f"  Thời gian: {stats['processing_time']:.2f}s")
        print(f"{'='*60}\n")


# ============================================================
# ENTRY POINT
# ============================================================
def run(image_path, debug=False, save=True, output_dir=None,
        answer_key=None, grading_method='standard'):
    """API chạy nhanh cho 1 ảnh: nhận diện, chấm điểm, lưu output.

    Args:
        image_path: Đường dẫn ảnh phiếu.
        debug: Bật log chi tiết.
        save: Có lưu ảnh/file kết quả hay không.
        output_dir: Thư mục output; None thì dùng thư mục mặc định.
        answer_key: File đáp án chuẩn để chấm điểm (nếu có).
    """
    omr = SmartOMR(debug=debug)
    if grading_method == 'crop_clean':
        result = omr.process_crop_clean(image_path)
    else:
        result = omr.process(image_path)
    if not result:
        return result

    # --- Cham diem neu co dap an ---
    grading_result = None
    if answer_key and GRADER_AVAILABLE:
        if os.path.isfile(answer_key):
            key_data = load_answer_key(answer_key)
            grading_result = grade(result['answers'], key_data)
            grading_result.print_report()
            result['grading'] = grading_result.to_dict()
        else:
            print(f"  [WARN] Khong tim thay file dap an: {answer_key}")

    if not save:
        return result

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(image_path))[0]

    # Luu anh annotated (co cham diem neu co)
    if grading_result is not None and GRADER_AVAILABLE:
        ann_img = draw_graded_annotated(
            result['image_orig'], result['grid'],
            result['answers'], grading_result)
        cv2.imwrite(os.path.join(output_dir, f"{base}_graded.jpg"), ann_img)
    cv2.imwrite(os.path.join(output_dir, f"{base}_annotated.jpg"), result['annotated'])

    # Luu anh tung cau
    q_dir = os.path.join(output_dir, f"{base}_questions")
    os.makedirs(q_dir, exist_ok=True)
    if 'question_images' in result:
        for q_num, qd in result['question_images'].items():
            ans = qd['answer'] if qd.get('answer') else '_'
            cv2.imwrite(os.path.join(q_dir, f"Q{q_num:03d}_{ans}.jpg"), qd['image'])
    result['q_dir'] = q_dir

    # Luu file dap an nhan dien
    txt_path = os.path.join(output_dir, f"{base}_answers.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"SmartOMR v3\nAnh: {image_path}\n{'='*40}\n\n")
        for q in sorted(result['answers'].keys()):
            a = result['answers'][q]
            f.write(f"Cau {q:3d}: {a if a else '-'}\n")
        s = result['stats']
        f.write(f"\n{'='*40}\n{s['answered']}/{s['total_questions']} cau co dap an\n")
        f.write(f"Thoi gian: {s['processing_time']:.2f}s\n")

    # Luu bao cao cham diem
    if grading_result is not None:
        report_path = os.path.join(output_dir, f"{base}_grade_report.txt")
        grading_result.save_report(report_path)

    print(f"  Da luu:")
    print(f"    {os.path.join(output_dir, base + '_annotated.jpg')}")
    if grading_result is not None:
        print(f"    {os.path.join(output_dir, base + '_graded.jpg')}")
        print(f"    {os.path.join(output_dir, base + '_grade_report.txt')}")
    print(f"    {txt_path}")
    print(f"    {q_dir}/ ({len(result['question_images'])} anh)")

    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='SmartOMR v3')
    p.add_argument('--image', '-i', required=True)
    p.add_argument('--debug', '-d', action='store_true')
    p.add_argument('--show', action='store_true')
    p.add_argument('--no-save', action='store_true')
    p.add_argument('--answer-key', '-k', default=None,
                   help='File dap an chuan (.txt/.json) de cham diem')
    p.add_argument('--create-key', action='store_true',
                   help='Tao file dap an mau (120 cau) tai answer_keys/template.txt')
    args = p.parse_args()

    if args.create_key:
        if GRADER_AVAILABLE:
            from modules.grader import create_template
            out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'answer_keys', 'template.txt')
            create_template(out, n_questions=NUM_QUESTIONS)
            print(f"  Tao file dap an mau: {out}")
            print(f"  Hay mo file va dien dap an A/B/C/D cho tung cau.")
        else:
            print("[ERROR] Khong load duoc modules/grader.py")
        sys.exit(0)

    if not os.path.isfile(args.image):
        print(f"Khong tim thay: {args.image}")
        sys.exit(1)

    result = run(args.image, debug=args.debug, save=not args.no_save,
                 answer_key=args.answer_key)

    if result and args.show:
        disp = result['annotated']
        scale = min(900 / disp.shape[0], 1.0)
        if scale < 1:
            disp = cv2.resize(disp, None, fx=scale, fy=scale)
        cv2.imshow("SmartOMR v3", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
