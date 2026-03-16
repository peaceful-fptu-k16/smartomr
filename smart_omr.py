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
    from modules.ml_grader import MLGrader
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

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

# Auto-scale: resize image to this target width for consistent detection
AUTO_SCALE_TARGET_W = 2500


class SmartOMR:
    def __init__(self, debug=False, ml_model_path=None, ml_mode="sum"):
        self.debug = debug
        self.ml_grader = None
        self._gray_clahe_cache = None
        if ml_model_path and ML_AVAILABLE:
            self.ml_grader = MLGrader(ml_model_path, mode=ml_mode)
            if not self.ml_grader.is_ready():
                print("  [WARN] ML model không load được, dùng threshold grading")
                self.ml_grader = None

    def process(self, image_path):
        print(f"\n{'='*60}")
        print(f"  SmartOMR v3")
        print(f"  Ảnh: {image_path}")
        print(f"{'='*60}")

        t0 = time.time()
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Không đọc được: {image_path}")
            return None

        # Auto-scale: resize small images for consistent HoughCircles detection
        h_orig, w_orig = image.shape[:2]
        self._scale_factor = 1.0
        if w_orig < AUTO_SCALE_TARGET_W * 0.75:
            self._scale_factor = AUTO_SCALE_TARGET_W / w_orig
            new_w = AUTO_SCALE_TARGET_W
            new_h = int(h_orig * self._scale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            print(f"  Auto-scaled: {w_orig}x{h_orig} → {new_w}x{new_h} (×{self._scale_factor:.2f})")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        print(f"  Kích thước: {w}x{h}")

        # Perspective correction: detect corner markers and warp
        warped = self._perspective_correct(image, gray)
        if warped is not None:
            image = warped
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            print(f"  Perspective corrected: {w}x{h}")

        # Auto-scale AFTER perspective correction to ~2500px (both up and down)
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

        # CLAHE-enhanced gray (cache dùng chung cho _detect_circles & _remove_header_rows)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._gray_clahe_cache = clahe.apply(gray)

        # === BƯỚC 1: Phát hiện tất cả circles ===
        print(f"\n[1/6] HoughCircles...")
        raw_circles = self._detect_circles(gray)
        print(f"  -> {len(raw_circles)} circles")

        # === BƯỚC 2: Lọc radius, bỏ outlier ===
        print(f"[2/6] Lọc theo radius...")
        good_circles = self._filter_by_radius(raw_circles, gray)
        print(f"  -> {len(good_circles)} circles (radius lọc)")

        # === BƯỚC 3: Phân 4 cột chính theo X ===
        print(f"[3/6] Chia 4 cột chính...")
        columns = self._split_into_main_columns(good_circles)
        for ci, col in enumerate(columns):
            if col:
                xs = [c[0] for c in col]
                print(f"  Cột {ci+1}: {len(col)} circles, x=[{min(xs)},{max(xs)}]")

        # === BƯỚC 4: Trong mỗi cột, phân hàng + ABCD ===
        print(f"[4/6] Phân hàng & xác định A/B/C/D...")
        grid = self._build_answer_grid(columns, gray)
        total_q = sum(len(rows) for rows in grid.values())
        print(f"  -> {total_q} câu nhận diện")

        # === BƯỚC 5: Grading (threshold) ===
        print(f"[5/7] Xác định đáp án (threshold)...")
        answers = self._grade_all(gray, grid)
        n_answered = sum(1 for v in answers.values() if v is not None)
        print(f"  -> {n_answered}/{len(answers)} câu có đáp án")

        # === BƯỚC 6: Cắt ảnh từng câu ===
        print(f"[6/7] Cắt ảnh câu hỏi...")
        question_images = self._crop_all_questions(image, gray, grid, answers)
        print(f"  -> {len(question_images)} ảnh")

        # === BƯỚC 7: ML Grading (nếu có model) ===
        ml_answers = {}
        if self.ml_grader and self.ml_grader.is_ready():
            print(f"[7/7] ML Grading...")
            ml_answers = self._ml_grade_all(question_images)
            n_ml = sum(1 for v in ml_answers.values() if v[0] is not None)
            print(f"  -> ML: {n_ml}/{len(ml_answers)} câu")
            # Kết hợp: ML override khi threshold không chắc chắn
            answers = self._combine_answers(answers, ml_answers, gray, grid)
            n_answered = sum(1 for v in answers.values() if v is not None)
            print(f"  -> Kết hợp: {n_answered}/{len(answers)} câu có đáp án")
        else:
            if self.ml_grader is None and self.debug:
                print(f"[7/7] ML Grading: bỏ qua (không có model)")

        annotated = self._draw_annotated(image, grid, answers)
        elapsed = time.time() - t0

        stats = {
            'total_circles': len(raw_circles),
            'filtered_circles': len(good_circles),
            'total_questions': NUM_QUESTIONS,
            'detected_questions': total_q,
            'answered': n_answered,
            'unanswered': total_q - n_answered,
            'processing_time': elapsed,
            'ml_enabled': self.ml_grader is not None,
            'ml_answers': ml_answers
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
            'image_orig': image
        }

    # ===================================================================
    # PERSPECTIVE CORRECTION using corner markers
    # ===================================================================
    def _perspective_correct(self, image, gray):
        """
        Detect 4 corner markers (■) of the answer grid and apply
        perspective warp to produce a standardized rectangle.
        Returns warped image, or None if markers not found.
        """
        h, w = gray.shape
        
        # Adaptive threshold for marker detection
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 21, 10)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for square-ish dark markers
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
        
        # Cluster nearby markers (same marker detected at multiple scales)
        clustered = self._cluster_markers(markers)
        
        if len(clustered) < 3:
            return None
        
        # Find 4 corner markers: the ones closest to image corners
        corners = self._find_corner_markers(clustered, w, h)
        
        # If 4 corners not found, try 3-corner estimation (parallelogram)
        if corners is None and len(clustered) >= 3:
            corners = self._estimate_4th_corner(clustered, w, h)
        
        if corners is None:
            return None
        
        tl, tr, bl, br = corners
        
        # Verify the quadrilateral makes sense
        quad_w = max(np.linalg.norm(np.array(tr) - np.array(tl)),
                     np.linalg.norm(np.array(br) - np.array(bl)))
        quad_h = max(np.linalg.norm(np.array(bl) - np.array(tl)),
                     np.linalg.norm(np.array(br) - np.array(tr)))
        
        # Quadrilateral should be reasonably sized (at least 40% of image)
        if quad_w < w * 0.35 or quad_h < h * 0.35:
            return None
        
        # Destination: standard rectangle with some padding
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
        """Merge markers that are very close (same marker detected at different scales)."""
        markers = sorted(markers, key=lambda m: m[2], reverse=True)  # sort by area, largest first
        used = [False] * len(markers)
        clustered = []
        
        for i, (cx, cy, area) in enumerate(markers):
            if used[i]:
                continue
            used[i] = True 
            # Mark nearby markers as used
            for j in range(i + 1, len(markers)):
                if not used[j]:
                    dx = abs(markers[j][0] - cx)
                    dy = abs(markers[j][1] - cy)
                    if dx < dist_threshold and dy < dist_threshold:
                        used[j] = True
            clustered.append((cx, cy, area))
        
        return clustered
    
    def _find_corner_markers(self, markers, img_w, img_h):
        """
        From a list of clustered markers, find the 4 that form the 
        corners of the answer grid area.
        Strategy: corner markers are typically the largest. Take the top 
        candidates by area and pick the 4 forming the best rectangle.
        """
        # Sort by area descending (corner markers are the largest on the sheet)
        markers_sorted = sorted(markers, key=lambda m: m[2], reverse=True)
        
        # Take top candidates (at most 10)
        candidates = markers_sorted[:min(10, len(markers_sorted))]
        
        # Try subsets starting from the 4 largest, expanding if needed
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
            
            # Check: top pair should have similar y, bottom pair similar y
            top_dy = abs(tl[1] - tr[1])
            bot_dy = abs(bl[1] - br[1])
            if top_dy > img_h * 0.15 or bot_dy > img_h * 0.15:
                continue
            
            # Width/height should be reasonable
            w1 = np.linalg.norm(np.array(tr) - np.array(tl))
            w2 = np.linalg.norm(np.array(br) - np.array(bl))
            h1 = np.linalg.norm(np.array(bl) - np.array(tl))
            h2 = np.linalg.norm(np.array(br) - np.array(tr))
            
            if min(w1, w2) < img_w * 0.3 or min(h1, h2) < img_h * 0.3:
                continue
            
            # Score: prefer larger area + more rectangular (parallel sides)
            area = 0.5 * abs(
                (tr[0] - tl[0]) * (br[1] - tl[1]) - (br[0] - tl[0]) * (tr[1] - tl[1]) +
                (br[0] - tr[0]) * (bl[1] - tr[1]) - (bl[0] - tr[0]) * (br[1] - tr[1])
            )
            w_ratio = min(w1, w2) / max(w1, w2) if max(w1, w2) > 0 else 0
            h_ratio = min(h1, h2) / max(h1, h2) if max(h1, h2) > 0 else 0
            rectangularity = w_ratio * h_ratio  # closer to 1 = more rectangular
            
            # Also prefer combos with larger markers (sum of areas)
            area_sum = sum(candidates[i][2] for i in combo)
            
            score = area * rectangularity * area_sum
            
            if score > best_score:
                best_score = score
                best_corners = (tl, tr, bl, br)
        
        return best_corners

    def _estimate_4th_corner(self, markers, img_w, img_h):
        """
        When only 3 corner markers are found, estimate the missing 4th
        using parallelogram properties: missing = p1 + p3 - p2 (diagonally opposite).
        """
        from itertools import combinations
        
        markers_sorted = sorted(markers, key=lambda m: m[2], reverse=True)
        candidates = markers_sorted[:min(10, len(markers_sorted))]
        
        best_score = -1
        best_corners = None
        
        for combo in combinations(range(len(candidates)), 3):
            pts = [(candidates[i][0], candidates[i][1]) for i in combo]
            
            # Try each of 4 roles as the missing corner
            for missing_role in ['tl', 'tr', 'bl', 'br']:
                # Assign 3 points to the other 3 roles
                # Sort remaining 3 by position to figure out assignment
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
                
                # Validate: estimated corner should be within image bounds (with margin)
                for p in [tl, tr, bl, br]:
                    if p[0] < -img_w * 0.1 or p[0] > img_w * 1.1:
                        break
                    if p[1] < -img_h * 0.1 or p[1] > img_h * 1.1:
                        break
                else:
                    # Validate dimensions
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
    # BƯỚC 1: Phát hiện circles
    # ===================================================================
    def _detect_circles(self, gray):
        # Dùng CLAHE đã cache (tránh tính lại)
        enhanced = self._gray_clahe_cache if self._gray_clahe_cache is not None else gray
        blurred = cv2.GaussianBlur(enhanced, (9, 9), 2)
        
        # Primary detection
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
        
        # Supplementary detection with relaxed params if too few circles found
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
                    # Only add if not too close to existing circle
                    if not any(abs(x-ex) < 15 and abs(y-ey) < 15 
                              for ex, ey in existing):
                        circles.append((x, y, r))
                        existing.add((x, y))
        return circles

    # ===================================================================
    # BƯỚC 2: Lọc radius + circularity
    # ===================================================================
    def _filter_by_radius(self, circles, gray=None):
        if not circles:
            return []
        radii = [c[2] for c in circles]
        med = np.median(radii)
        filtered = [(x, y, r) for x, y, r in circles if abs(r - med) < med * 0.35]
        
        # Additional filter: reject square markers and noise
        # Check that each circle has a ring-like pattern (bright center or ring edge)
        if gray is not None and len(filtered) > 20:
            validated = []
            h, w = gray.shape
            for x, y, r in filtered:
                # Get bounding box
                x1, y1 = max(0, x - r - 2), max(0, y - r - 2)
                x2, y2 = min(w, x + r + 3), min(h, y + r + 3)
                crop = gray[y1:y2, x1:x2]
                if crop.size < 10:
                    continue
                
                # Check circularity via contour analysis on local crop
                _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(cnt)
                    perim = cv2.arcLength(cnt, True)
                    if perim > 0:
                        circularity = 4 * np.pi * area / (perim * perim)
                        # Real circles: circularity > 0.5; squares/irregular < 0.5
                        if circularity < 0.35:
                            continue
                
                validated.append((x, y, r))
            
            if len(validated) >= len(filtered) * 0.6:  # don't filter too aggressively
                return validated
        
        return filtered

    # ===================================================================
    # BƯỚC 3: Chia 4 cột chính qua sub-column clustering
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
        
        Returns: {col_idx: [row0, row1, ...]}
                 Mỗi row = [(cx,cy,r), ...] đã sort trái→phải (chỉ 4 circles ABCD)
        """
        grid = {}

        # Trước tiên, xác định 4 subcol X centers cho mỗi cột chính
        # Scan ALL columns first to find the best reference spacing
        ref_subcol_spacings = []
        ref_y_starts = []
        ref_y_ends = []
        
        for col_idx in range(len(columns)):
            col_circles = columns[col_idx]
            if len(col_circles) < 20:
                continue
            subcols = self._cluster_x_local(col_circles, threshold=30)
            # Merge very close subcols
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
                # Sort by center X, take first 4 (or best 4)
                sc_sorted = sorted(valid_sc, key=lambda sc: np.mean([c[0] for c in sc]))[:4]
                centers = [np.mean([c[0] for c in sc]) for sc in sc_sorted]
                spacings = [centers[i+1] - centers[i] for i in range(3)]
                # Only use if spacing is fairly regular (max/min < 1.5)
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
            print(f"    Answer Y range: [{y_min_answer:.0f}, {y_max_answer:.0f}]")

        # Compute ABCD width from Col 1-3 for boundary correction
        ref_abcd_d_positions = []  # rightmost ABCD subcol (D) x for cols 1-3
        ref_abcd_widths = []
        for col_idx in range(min(3, len(columns))):
            rows_c = grid.get(col_idx, [])
            if not rows_c:
                continue
            # Get subcol centers from 4-circle rows
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
            
            # Boundary correction: check if Col 4 is missing subcol A
            # by comparing its ABCD width to reference columns
            if col_idx == 3 and ref_abcd_widths and col_idx - 1 in grid:
                col4_subcols = self._cluster_x_local(filtered, threshold=20) if filtered else []
                col4_valid = sorted([sc for sc in col4_subcols if len(sc) >= 15],
                                   key=lambda sc: np.mean([c[0] for c in sc]))
                if len(col4_valid) >= 3:
                    col4_width = np.mean([c[0] for c in col4_valid[-1]]) - np.mean([c[0] for c in col4_valid[0]])
                    if col4_width < expected_abcd_width * 0.85:
                        # Col 4 is too narrow — likely missing leftmost subcol
                        # Look for circles in Col 3 that should belong to Col 4
                        col3_circles = columns[col_idx - 1]
                        col4_first_x = np.mean([c[0] for c in col4_valid[0]])
                        expected_a_x = col4_first_x - avg_subcol_spacing
                        # Steal circles from col3 that are near expected_a_x
                        stolen = []
                        remaining_col3 = []
                        for c in col3_circles:
                            if abs(c[0] - expected_a_x) < avg_subcol_spacing * 0.4:
                                stolen.append(c)
                            else:
                                remaining_col3.append(c)
                        if len(stolen) >= 10:
                            # Also steal nearby noise subcol circles
                            stolen_y = [(x, y, r) for x, y, r in stolen
                                       if y_min_answer <= y <= y_max_answer]
                            filtered = stolen_y + filtered
                            columns[col_idx - 1] = remaining_col3
                            if self.debug:
                                print(f"    Boundary fix: moved {len(stolen_y)} circles from Col3 to Col4")
                            # Re-process Col 3 with corrected circles
                            grid[col_idx - 1] = self._process_one_column(
                                col_idx - 1, remaining_col3, gray, avg_subcol_spacing)
            
            if self.debug:
                print(f"    Cột {col_idx+1}: {len(col_circles)} → {len(filtered)} circles (Y filter)")
            
            rows = self._process_one_column(col_idx, filtered, gray, avg_subcol_spacing)
            grid[col_idx] = rows

        return grid

    def _process_one_column(self, col_idx, col_circles, gray, avg_subcol_spacing):
        """Xử lý 1 cột: extract ABCD → cluster Y → remove header → validate."""
        valid_rows = self._process_one_column_inner(col_idx, col_circles, gray, avg_subcol_spacing)
        
        # Fallback: if too few rows detected, retry with relaxed subcol threshold
        # (handles images with imprecise perspective where circles scatter more)
        if len(valid_rows) < QUESTIONS_PER_COLUMN * 0.85:
            retry = self._process_one_column_inner(col_idx, col_circles, gray, avg_subcol_spacing, subcol_threshold=30)
            if len(retry) > len(valid_rows):
                if self.debug:
                    print(f"    Cột {col_idx+1}: fallback {len(valid_rows)}→{len(retry)} rows (relaxed threshold)")
                valid_rows = retry
        
        return valid_rows

    def _process_one_column_inner(self, col_idx, col_circles, gray, avg_subcol_spacing, subcol_threshold=20):
        """Inner processing for one column with configurable subcol threshold."""
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

        # Nếu nhiều hơn 30 hàng → bỏ từ ĐẦU trước (header luôn nằm trên cùng)
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
        """Cluster circles by X within a column."""
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
        Strategy:
        1. Subcol clustering with threshold ~25px
        2. Filter noise subcols (too few circles)
        3. If 4+ valid subcols: pick best 4 by spacing regularity
        4. Fallback to k-means with noise rejection
        """
        if len(col_circles) < 8:
            return col_circles
            
        # Strategy 1: subcol clustering with configurable threshold
        subcols = self._cluster_x_local(col_circles, threshold=threshold)
        
        # Merge subcols that are very close (< 20% of avg_spacing)
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

        # Filter: keep subcols with >= 15 circles (real bubble columns ≈ 30)
        valid_subcols = [sc for sc in subcols if len(sc) >= 15]
        
        if len(valid_subcols) == 4:
            result = []
            for sc in valid_subcols:
                result.extend(sc)
            return result
        
        if len(valid_subcols) > 4:
            # Pick best 4 by spacing regularity
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
            # Missing one subcol. Use spacing from the 3 we have to predict the 4th.
            centers3 = sorted([np.mean([c[0] for c in sc]) for sc in valid_subcols])
            spacings = [centers3[i+1] - centers3[i] for i in range(len(centers3)-1)]
            sp = np.median(spacings) if spacings else avg_spacing
            
            # Determine which position is missing: check if gap exists at edges
            # If spacing between 0-1 is ~2*sp → missing between them
            # If first center - column_left_edge >> sp → missing at start
            # If column_right_edge - last center >> sp → missing at end
            all_xs = [c[0] for c in col_circles]
            x_min, x_max = min(all_xs), max(all_xs)
            
            # Check for double gap
            found_double = False
            for i in range(len(spacings)):
                if spacings[i] > sp * 1.5:
                    # Gap between centers3[i] and centers3[i+1] → insert center
                    mid = (centers3[i] + centers3[i+1]) / 2
                    # Create a fake 4th subcol center
                    centers4 = sorted(centers3 + [mid])
                    found_double = True
                    break
            
            if not found_double:
                # Check edges
                left_gap = centers3[0] - x_min
                right_gap = x_max - centers3[-1]
                if left_gap > sp * 0.5:
                    centers4 = sorted([centers3[0] - sp] + centers3)
                elif right_gap > sp * 0.5:
                    centers4 = sorted(centers3 + [centers3[-1] + sp])
                else:
                    centers4 = centers3
            
            # Assign circles to nearest of the 4 centers
            if len(centers4) == 4:
                half = sp * 0.6
                result = []
                for c in col_circles:
                    dists = [abs(c[0] - ctr) for ctr in centers4]
                    if min(dists) < half:
                        result.append(c)
                return result
        
        # Fallback: return all circles in the column
        return col_circles

    def _cluster_y(self, circles):
        """Gom circles thành hàng dựa trên Y."""
        sorted_c = sorted(circles, key=lambda c: c[1])
        
        ys = [c[1] for c in sorted_c]
        diffs = [ys[i+1] - ys[i] for i in range(len(ys)-1)]
        if not diffs:
            return [sorted_c]

        # Find the first significant gap in sorted diffs to separate
        # within-row diffs (~0-10px) from between-row diffs (30+px).
        # Using "first gap > 20" is robust against outlier circles that
        # create very large diffs which would skew a "largest gap" approach.
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
    # BƯỚC 5: Grading (tối ưu: dùng ROI crop thay vì full-image mask)
    # ===================================================================
    @staticmethod
    def _bubble_contrast(gray, cx, cy, r):
        """
        Tính inner_mean và outer_mean cho 1 bubble bằng ROI crop nhỏ.
        Nhanh hơn ~100x so với tạo np.zeros(gray.shape) full-image.
        Returns: (inner_val, outer_val, contrast_ratio)
        """
        cx, cy, r = int(cx), int(cy), int(r)
        h, w = gray.shape
        outer_r = min(int(r * 2.0), r + 30)
        # Use small inner radius (r*0.4) to sample only the TRUE center
        # of the bubble, avoiding the printed circle outline which is dark
        # even on unfilled bubbles and causes false positives.
        inner_r = max(int(r * 0.4), 3)
        outer_inner = max(int(r * 1.3), r + 3)

        # ROI bounding box cho outer ring
        x1 = max(0, cx - outer_r)
        y1 = max(0, cy - outer_r)
        x2 = min(w, cx + outer_r + 1)
        y2 = min(h, cy + outer_r + 1)
        roi = gray[y1:y2, x1:x2]
        if roi.size < 4:
            return 200.0, 200.0, 1.0

        rh, rw = roi.shape
        lcx, lcy = cx - x1, cy - y1  # local center

        # Inner mask (small ROI)
        inner_mask = np.zeros((rh, rw), dtype="uint8")
        cv2.circle(inner_mask, (lcx, lcy), inner_r, 255, -1)
        inner_val = cv2.mean(roi, mask=inner_mask)[0]

        # Outer annular ring mask (same small ROI)
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
        
        Real pencil fills: fr80 >= 25% (đa số pixel < 80)
        Mild contamination (border/shadow): fr80 ~ 0% (pixel ~100-140, không đủ tối)
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
        """Grade all questions using per-row relative analysis.
        
        Within each row of 4 bubbles (ABCD), find the darkest one.
        A filled bubble is detected when:
        1. Its contrast ratio (inner/outer) is low (< 0.65)
        2. It's significantly darker than the next-darkest bubble (gap >= 40)
        
        Also recovers missing circles: when a row has only 3 circles
        and none are filled, check the missing subcol position for a dark region.
        """
        answers = {}
        q_num = 1
        for col_idx in sorted(grid.keys()):
            col_rows = grid[col_idx]
            
            # Compute expected subcol X centers from rows with 4 circles
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
                
                # Recovery: if 3 circles, no answer found, try the missing position
                if answer is None and n == 3 and all(x is not None for x in expected_xs):
                    row_xs = [row[ci][0] for ci in range(3)]
                    row_y = int(np.mean([row[ci][1] for ci in range(3)]))
                    r = int(avg_r)
                    
                    # Find which subcol is missing
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
                                print(f"    Q{q_num:3d}: recovered {answer} at missing subcol (iv={best_iv:.0f}, cr={best_cr:.2f}, dx={best_dx})")
                
                answers[q_num] = answer
                
                if self.debug:
                    a = answer if answer else "-"
                    ivs = ', '.join(f'{v:.0f}' for v in inner_vals)
                    cs = ', '.join(f'{c:.2f}' for c in contrasts)
                    print(f"    Q{q_num:3d}: {a}  inner=[{ivs}] cr=[{cs}]")
                q_num += 1
        return answers

    def _grade_row(self, inner_vals, contrasts, gray=None, row=None):
        """Grade a single row by finding the darkest bubble relative to others."""
        n = min(len(inner_vals), NUM_CHOICES)
        if n < 3:
            return None
        
        # Find darkest bubble
        vals = inner_vals[:n]
        sorted_vals = sorted(vals)
        min_val = sorted_vals[0]
        second_val = sorted_vals[1]
        min_idx = vals.index(min_val)
        
        # Absolute brightness check: if the darkest bubble is still very bright,
        # it's definitely not filled (empty bubble inner ~190-220)
        if min_val > 185:
            return None
        
        # Gap between darkest and 2nd darkest
        gap = second_val - min_val
        
        # Criteria for a filled bubble:
        # 1. Must be significantly darker than the next bubble
        if gap < 40:
            return None
        
        # 2. For borderline inner values (90-185): verify with fill ratio
        #    Real fills have many pixels < 80 (fr80 >= 25%)
        #    Mild contamination (shadows, borders) has inner 90-185 but fr80 ~ 0%
        if min_val >= 90 and gray is not None and row is not None and min_idx < len(row):
            cx, cy, r = row[min_idx]
            fr = self._bubble_fill_ratio(gray, cx, cy, r, threshold=80)
            if fr < 0.20:
                return None
        
        # 3. Contrast ratio must be low (inner much darker than outer ring)
        #    Tier 1: normal fill (strong evidence)
        if contrasts[min_idx] < 0.72:
            return CHOICE_LABELS[min_idx]
        
        #    Tier 2: light fill — needs stronger evidence
        #    Inner must be noticeably dark (<160) and contrast clearly below paper
        if contrasts[min_idx] < 0.82 and min_val < 160 and gap >= 50:
            return CHOICE_LABELS[min_idx]
        
        return None

    # ===================================================================
    # CONTAMINATION DETECTION
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
    # BƯỚC 7: ML Grading
    # ===================================================================
    def _ml_grade_all(self, question_images):
        """Dùng ML model để grade tất cả câu hỏi — batch predict (nhanh hơn ~50x)."""
        crops = {q_num: qi['image'] for q_num, qi in question_images.items()}
        return self.ml_grader.predict_batch(crops)

    def _combine_answers(self, threshold_answers, ml_answers, gray, grid):
        """Kết hợp kết quả threshold và ML.
        
        Chiến lược:
        - Nếu cả 2 đồng ý → dùng kết quả chung
        - Nếu threshold = None, ML có câu trả lời → dùng ML
        - Nếu khác nhau → dùng ML nếu confidence > 0.7, ngược lại dùng threshold
        """
        combined = dict(threshold_answers)
        for q_num, (ml_ans, ml_conf) in ml_answers.items():
            th_ans = threshold_answers.get(q_num)
            
            if th_ans == ml_ans:
                continue  # Đồng ý
            
            # ML predict 'blank' → coi như None
            effective_ml = ml_ans if ml_ans != 'blank' else None
            
            if th_ans is None and effective_ml is not None and ml_conf > 0.85:
                combined[q_num] = effective_ml
                if self.debug:
                    print(f"  Q{q_num:03d}: threshold=None -> ML={ml_ans} (conf={ml_conf:.2f})")
            elif th_ans is not None and effective_ml is None and ml_conf > 0.85:
                # ML nói blank với confidence cao → bỏ trống
                combined[q_num] = None
                if self.debug:
                    print(f"  Q{q_num:03d}: threshold={th_ans} -> ML=blank (conf={ml_conf:.2f})")
            elif th_ans != effective_ml and effective_ml is not None and ml_conf > 0.85:
                combined[q_num] = effective_ml
                if self.debug:
                    print(f"  Q{q_num:03d}: threshold={th_ans} -> ML={ml_ans} (conf={ml_conf:.2f})")
        
        return combined

    # ===================================================================
    # BƯỚC 6: Cắt ảnh từng câu
    # ===================================================================
    def _crop_all_questions(self, image, gray, grid, answers):
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

                # Thêm vùng số thứ tự bên trái
                bubble_width = max_x - min_x
                label_extra = int(bubble_width * 0.40)

                x1 = max(0, min_x - label_extra - CROP_PAD_X)
                x2 = min(w, max_x + CROP_PAD_X)
                y1 = max(0, min_y - CROP_PAD_Y)
                y2 = min(h, max_y + CROP_PAD_Y)

                crop = image[y1:y2, x1:x2].copy()

                # Đánh dấu đáp án
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
        ml_model=None, ml_mode="raw", answer_key=None):
    omr = SmartOMR(debug=debug, ml_model_path=ml_model, ml_mode=ml_mode)
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
    for q_num, qd in result['question_images'].items():
        ans = qd['answer'] if qd['answer'] else '_'
        cv2.imwrite(os.path.join(q_dir, f"Q{q_num:03d}_{ans}.jpg"), qd['image'])

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
    p.add_argument('--ml-model', '-m', help='Path to trained ML model (.pkl)')
    p.add_argument('--ml-mode', choices=['raw', 'sum', 'pixel'], default='raw',
                   help='Feature mode: raw (best), sum (60 features) or pixel (900 features)')
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
                 ml_model=args.ml_model, ml_mode=args.ml_mode,
                 answer_key=args.answer_key)

    if result and args.show:
        disp = result['annotated']
        scale = min(900 / disp.shape[0], 1.0)
        if scale < 1:
            disp = cv2.resize(disp, None, fx=scale, fy=scale)
        cv2.imshow("SmartOMR v3", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
