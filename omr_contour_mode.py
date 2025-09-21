# Grid-free OMR using contour detection + robust clustering and conservative selection.

import cv2
import numpy as np
from typing import List, Tuple, Dict
import collections

SubjectNames = ["PYTHON","DATA ANALYSIS","MySQL","POWER BI","Adv STATS"]

# ---------- Preprocess ----------

def preprocess_for_contours(image_bgr: np.ndarray, warp_w: int, warp_h: int) -> Tuple[np.ndarray, np.ndarray]:
    warped = cv2.resize(image_bgr, (warp_w, warp_h), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 25, 25)

    # Dynamic thresholding based on image mean
    mean_val = gray.mean()
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10 + int(mean_val / 20)
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary, warped

# ---------- Candidates ----------

def find_bubble_candidates(binary: np.ndarray, warped: np.ndarray) -> List[Tuple[np.ndarray, float, float, float]]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    
    # 1. Contour-based detection
    for c in contours:
        area = cv2.contourArea(c)
        if area < 35 or area > 2200:
            continue
        x,y,w,h = cv2.boundingRect(c)
        if w < 7 or h < 7 or w > 36 or h > 36:
            continue
        ar = w / (h + 1e-6)
        peri = cv2.arcLength(c, True)
        circ = 4.0 * np.pi * area / (peri*peri + 1e-6)
        if 0.5 <= ar <= 1.8 and 0.3 <= circ <= 1.3:
            cands.append((c, area, circ, ar))
    
    # 2. Hybrid approach: Add Hough circles if contour detection is sparse
    if len(cands) < 200:
        circles = cv2.HoughCircles(
            cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), (7,7), 0),
            cv2.HOUGH_GRADIENT, dp=1.2, minDist=14,
            param1=80, param2=18, minRadius=7, maxRadius=18
        )
        if circles is not None:
            H, W = binary.shape[:2]
            for (cx, cy, r) in np.round(circles[0, :]).astype(int):
                if 0 <= cx < W and 0 <= cy < H:
                    cnt = cv2.ellipse2Poly((cx, cy), (r, r), 0, 0, 360, 10)
                    cands.append((cnt.reshape(-1,1,2), float(np.pi*r*r), 1.0, 1.0))
    
    # Optional: consolidate overlapping candidates from the two methods
    # For simplicity, we'll just return the combined list.
    
    return cands

def contour_centers(cands) -> np.ndarray:
    pts = []
    for c,_,_,_ in cands:
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = M["m10"]/M["m00"]
            cy = M["m01"]/M["m00"]
            pts.append([cx, cy])
    return np.array(pts, dtype=np.float32)

# ---------- Robust 1D quantization ----------

def quantize_to_bins(values: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    N = values.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    Kp = min(K, N)
    data = values.astype(np.float32).reshape(-1,1)
    if Kp >= 2:
        compactness, labels, centers = cv2.kmeans(
            data, Kp, None,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1),
            5, cv2.KMEANS_PP_CENTERS
        )
        order = np.argsort(centers.flatten())
        remap = {orig:new for new,orig in enumerate(order)}
        labels = np.vectorize(remap.get)(labels.flatten())
        centers = centers.flatten()[order]
        return labels, centers
    else:
        return np.zeros((N,), dtype=np.int32), np.array([values.mean()], dtype=np.float32)

# ---------- Grid clustering (quantile columns) ----------

def cluster_grid(pts: np.ndarray, cols: int = 5, rows: int = 20, opts: int = 4):
    assignments = []
    xs = pts[:, 0]
    try:
        cuts = np.quantile(xs, [i/cols for i in range(1, cols)], method="linear")
    except TypeError:
        cuts = np.quantile(xs, [i/cols for i in range(1, cols)])
    col_labels = np.digitize(xs, cuts, right=False)

    for col in range(cols):
        idx_c = np.where(col_labels == col)[0]
        if len(idx_c) == 0:
            continue
        pts_c = pts[idx_c]
        ys = pts_c[:,1]
        y_min, y_max = float(ys.min()), float(ys.max())
        if y_max - y_min < 1e-3:
            band_edges = np.linspace(y_min, y_min + rows, rows+1)
        else:
            band_edges = np.linspace(y_min, y_max, rows+1)
        band_centers = 0.5 * (band_edges[:-1] + band_edges[1:])
        idx_to_row = np.abs(pts_c[:,1].reshape(-1,1) - band_centers.reshape(1,-1)).argmin(axis=1)

        for r in range(rows):
            idx_r = idx_c[np.where(idx_to_row == r)[0]]
            if len(idx_r) == 0:
                continue
            pts_r = pts[idx_r]
            order_idx = np.argsort(pts_r[:,0])
            ranks = np.empty_like(order_idx)
            ranks[order_idx] = np.arange(len(order_idx))
            if len(pts_r) < opts:
                scale = (opts - 1) / max(1, len(pts_r) - 1)
                x4_labels = np.round(ranks * scale).astype(int)
            else:
                bins = np.linspace(0, len(pts_r), opts+1)
                x4_labels = np.digitize(ranks, bins) - 1
                x4_labels = np.clip(x4_labels, 0, opts-1)
            for k, gi in enumerate(idx_r):
                assignments.append((gi, col, r, int(x4_labels[k])))
    return assignments

# ---------- Scoring ----------

def mask_fill_ratio(binary: np.ndarray, contour: np.ndarray) -> float:
    mask = np.zeros(binary.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.erode(mask, k, 1)
    mask = cv2.dilate(mask, k, 1)
    ink = cv2.countNonZero(cv2.bitwise_and(binary, mask))
    area = cv2.countNonZero(mask)
    return ink / max(1, area)

def evaluate_contour_mode(image_bgr: np.ndarray,
                          warp_w: int = 1000,
                          warp_h: int = 1400,
                          threshold: float = 0.16,
                          margin: float = 0.05):
    """
    Full pipeline: preprocess, detect contours, cluster to 5x20x4, score per contour mask.
    Returns (answers, scores, warped_bgr, binary).
    """
    binary, warped = preprocess_for_contours(image_bgr, warp_w, warp_h)
    cands = find_bubble_candidates(binary, warped)
    if not cands:
        return {}, {}, warped, binary
    
    pts = contour_centers(cands)
    assignments = cluster_grid(pts, cols=5, rows=20, opts=4)
    
    results: Dict[str, Dict[int, List[str]]] = {s: collections.defaultdict(list) for s in SubjectNames}
    scores: Dict[str, List[List[float]]] = {s:[[0,0,0,0] for _ in range(20)] for s in SubjectNames}
    
    for gi, col, row, opt in assignments:
        if not (0 <= col < len(SubjectNames)) or not (0 <= row < 20) or not (0 <= opt < 4):
            continue
        contour = cands[gi][0]
        ratio = mask_fill_ratio(binary, contour)
        
        # Store the bubble with its fill ratio
        results[SubjectNames[col]][row+1].append((opt, ratio))
    
    final_answers: Dict[str, Dict[int, str]] = {s:{} for s in SubjectNames}
    letters = "ABCD"
    
    # Robust score calculation: pick if it's the best and a significant margin exists
    for subject, q_data in results.items():
        for q_num, bubbles in q_data.items():
            if not bubbles:
                final_answers[subject][q_num] = ""
                continue
            
            # Sort bubbles by fill ratio in descending order
            bubbles.sort(key=lambda x: x[1], reverse=True)
            
            best_opt, best_ratio = bubbles[0]
            
            # Check for conservative selection: is the best ratio significantly higher?
            if len(bubbles) > 1:
                second_best_ratio = bubbles[1][1]
            else:
                second_best_ratio = 0
            
            # Apply thresholds: is the best ratio above the absolute threshold AND is it
            # also significantly higher than the next best?
            if best_ratio >= threshold and (best_ratio - second_best_ratio) > margin:
                final_answers[subject][q_num] = letters[best_opt]
            else:
                final_answers[subject][q_num] = ""
                
    return final_answers, scores, warped, binary
