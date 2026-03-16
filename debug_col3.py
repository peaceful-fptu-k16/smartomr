import cv2, numpy as np
from smart_omr import SmartOMR

omr = SmartOMR(debug=False)
img = cv2.imread('input/phieu_thi.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
warped = omr._perspective_correct(img, gray)
image = warped
h_cur, w_cur = image.shape[:2]
scale = 2500 / w_cur
image = cv2.resize(image, (2500, int(h_cur * scale)), interpolation=cv2.INTER_AREA)
gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
omr._gray_clahe_cache = clahe.apply(gray2)
raw = omr._detect_circles(gray2)
good = omr._filter_by_radius(raw, gray2)
columns = omr._split_into_main_columns(good)

# Col 3 (index 2) subcol analysis
col3 = columns[2]
print(f'Col 3 raw: {len(col3)} circles, x=[{min(c[0] for c in col3)},{max(c[0] for c in col3)}]')

subcols = omr._cluster_x_local(col3, threshold=20)
print(f'Subcols: {len(subcols)}')
for i, sc in enumerate(subcols):
    cx = np.mean([c[0] for c in sc])
    print(f'  SC{i}: {len(sc)} circles, cx={cx:.0f}, x=[{min(c[0] for c in sc)},{max(c[0] for c in sc)}]')

# After ABCD extraction
abcd = omr._extract_abcd_circles(col3, 104.8)
abcd_subcols = omr._cluster_x_local(abcd, threshold=20)
print(f'\nAfter ABCD extraction ({len(abcd)} circles):')
for i, sc in enumerate(abcd_subcols):
    cx = np.mean([c[0] for c in sc])
    print(f'  ABCD SC{i}: {len(sc)} circles, cx={cx:.0f}')

# Reference: Col 1, 2 ABCD centers
for ci in [0, 1]:
    col = columns[ci]
    scs = omr._cluster_x_local(col, threshold=20)
    valid = [sc for sc in scs if len(sc) >= 15]
    if len(valid) >= 4:
        valid.sort(key=lambda sc: np.mean([c[0] for c in sc]))
        centers = [np.mean([c[0] for c in sc]) for sc in valid[:4]]
        sp = [centers[i+1] - centers[i] for i in range(3)]
        print(f'Col {ci+1} ABCD: centers={[int(c) for c in centers]}, spacings={[int(s) for s in sp]}')
