"""Save annotated image showing detected circles for visual inspection."""
import cv2
import numpy as np
from smart_omr import SmartOMR, AUTO_SCALE_TARGET_W

for img_name in ['input/3.jpg', 'input/1.jpg']:
    img = cv2.imread(img_name)
    omr = SmartOMR(debug=False)
    gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    warped = omr._perspective_correct(img, gray_orig)
    if warped is not None:
        img = warped
    
    h_cur, w_cur = img.shape[:2]
    if w_cur < AUTO_SCALE_TARGET_W * 0.85:
        scale = AUTO_SCALE_TARGET_W / w_cur
        new_w = AUTO_SCALE_TARGET_W
        new_h = int(h_cur * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    omr._gray_clahe_cache = clahe.apply(gray)
    
    raw = omr._detect_circles(gray)
    filtered = omr._filter_by_radius(raw, gray)
    
    # Draw ALL detected circles on image
    vis = img.copy()
    for x, y, r in filtered:
        cv2.circle(vis, (int(x), int(y)), int(r), (0, 255, 0), 2)
        # Mark center
        cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
    
    base = img_name.split('/')[-1].split('.')[0]
    cv2.imwrite(f'output/debug_all_circles_{base}.jpg', vis)
    
    # Crop row 1 area from column 1 (wider to see context)
    columns = omr._split_into_main_columns(filtered)
    grid = omr._build_answer_grid(columns, gray)
    
    if grid.get(0) and len(grid[0]) > 0:
        row = grid[0][0]
        # Get bounding box with large padding
        xs = [c[0] for c in row]
        ys = [c[1] for c in row]
        rs = [c[2] for c in row]
        pad = 60
        x1 = max(0, int(min(xs) - max(rs) - pad))
        x2 = min(gray.shape[1], int(max(xs) + max(rs) + pad))
        y1 = max(0, int(min(ys) - max(rs) - pad))
        y2 = min(gray.shape[0], int(max(ys) + max(rs) + pad))
        
        crop = vis[y1:y2, x1:x2]
        cv2.imwrite(f'output/debug_row1_crop_{base}.jpg', crop)
        
        # Also save gray version
        gray_crop = gray[y1:y2, x1:x2]
        cv2.imwrite(f'output/debug_row1_gray_{base}.jpg', gray_crop)
        
        print(f'{base}: Row 1 crop saved [{x1}:{x2}, {y1}:{y2}]')
        
        # Check raw pixel scan across Y at center of row
        row_y = int(np.mean(ys))
        scan_line = gray[row_y, x1:x2]
        print(f'  Pixel scan at y={row_y}: min={scan_line.min()}, max={scan_line.max()}')
        
        # Find dark regions in scan line
        dark_positions = np.where(scan_line < 150)[0]
        if len(dark_positions) > 0:
            print(f'  Dark pixels (<150): {len(dark_positions)} positions')
            # Group consecutive dark pixels
            groups = []
            start = dark_positions[0]
            for i in range(1, len(dark_positions)):
                if dark_positions[i] - dark_positions[i-1] > 3:
                    groups.append((start + x1, dark_positions[i-1] + x1, dark_positions[i-1] - start + 1))
                    start = dark_positions[i]
            groups.append((start + x1, dark_positions[-1] + x1, dark_positions[-1] - start + 1))
            for gx1, gx2, width in groups:
                print(f'    Dark region: x=[{gx1},{gx2}] width={width}')
