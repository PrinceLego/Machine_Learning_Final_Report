

import os
import numpy as np
import cv2
from skimage import measure, morphology
from scipy.ndimage import gaussian_filter
import random
from tqdm import tqdm

# --- 1. 核心輔助函式 ---
def extract_roi(binary_map, area_min=25, dilate_radius=2, erode_radius=1):
    labels = measure.label(binary_map, connectivity=2)
    out_mask = np.zeros_like(binary_map, dtype=np.uint8)
    for region in measure.regionprops(labels):
        if region.area >= area_min:
            out_mask[labels == region.label] = 1
    se_dil = morphology.disk(dilate_radius)
    se_ero = morphology.disk(erode_radius)
    out_mask = morphology.binary_dilation(out_mask, se_dil)
    out_mask = morphology.binary_erosion(out_mask, se_ero)
    out_mask = out_mask.astype(np.uint8)
    coords = np.column_stack(np.where(out_mask > 0))
    if coords.size == 0:
        return out_mask, (out_mask.shape[1]//2, out_mask.shape[0]//2)
    yc = int(coords[:,0].mean()); xc = int(coords[:,1].mean())
    return out_mask, (xc, yc)

def compute_weight_map(gray_img, roi_mask, centroid, alpha=1.0, beta=1.0, sigma=26.0):
    lap = cv2.Laplacian(gray_img.astype(np.float32), cv2.CV_32F)
    lap = np.abs(lap)
    if lap.max() > 0:
        lap = lap / (lap.max() + 1e-9)
    h,w = gray_img.shape
    xv, yv = np.meshgrid(np.arange(w), np.arange(h))
    xc, yc = centroid
    dist_term = np.exp(-((xv - xc)**2 + (yv - yc)**2) / (2.0 * (sigma**2)))
    weight = alpha * lap + beta * dist_term
    weight *= roi_mask
    if weight.max() > 0:
        weight = weight / (weight.max() + 1e-9)
    return weight

def augment_heatmap(img, max_rotation=360, flip_prob=0.5, max_translate=4):
    h,w = img.shape
    angle = random.uniform(0, max_rotation)
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
    out = cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if random.random() < flip_prob:
        if random.random() < 0.5:
            out = np.fliplr(out)
        else:
            out = np.flipud(out)
    tx = random.randint(-max_translate, max_translate)
    ty = random.randint(-max_translate, max_translate)
    M2 = np.array([[1,0,tx],[0,1,ty]], dtype=np.float32)
    out = cv2.warpAffine(out, M2, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out

# --- 2. 合成函式，保持 float32 (之後轉 int32) ---
def merge_image_list(images_and_masks_to_merge, alpha=1.0, beta=1.0, sigma=26):
    if not images_and_masks_to_merge:
        return None
    h, w = images_and_masks_to_merge[0][0].shape
    H_sum = np.zeros((h, w), dtype=np.float32)

    for heatmap, mask in images_and_masks_to_merge:
        Haug = augment_heatmap(heatmap)
        roi_aug_float = augment_heatmap(mask.astype(np.float32))
        roi_aug_bin = (roi_aug_float > 0.5).astype(np.uint8)
        coords = np.column_stack(np.where(roi_aug_bin > 0))
        if coords.size == 0:
            yc, xc = h//2, w//2
        else:
            yc = int(coords[:,0].mean()); xc = int(coords[:,1].mean())
        Wi = compute_weight_map(Haug, roi_aug_bin, (xc, yc), alpha=alpha, beta=beta, sigma=sigma)
        H_sum += Wi * Haug

    merged = H_sum
    if merged.max() > 0:
        merged = merged / (merged.max() + 1e-9)  # 正規化到 0~1
    return merged.astype(np.float32)

# --- 3. 主程式 ---
if __name__ == "__main__":
    SINGLE_DEFECT_FILE = 'single_defect_wafers.npz'
    SYNTHETIC_OUTPUT_FILE = 'synthetic_mixed_dataset.npz'

    target_synthesis_list = {
        (0, 0, 0, 0, 1, 0, 1, 0): 1000,
        (0, 0, 0, 1, 0, 0, 1, 0): 1000,
        (0, 0, 0, 1, 1, 0, 0, 0): 1000,
        (0, 0, 0, 1, 1, 0, 1, 0): 1000,
        (0, 0, 1, 0, 0, 0, 1, 0): 1000,
        (0, 0, 1, 0, 1, 0, 0, 0): 1000,
        (0, 0, 1, 0, 1, 0, 1, 0): 1000,
        (0, 1, 0, 0, 0, 0, 1, 0): 1000,
        (0, 1, 0, 0, 1, 0, 0, 0): 1000,
        (0, 1, 0, 0, 1, 0, 1, 0): 1000,
        (0, 1, 0, 1, 0, 0, 0, 0): 1000,
        (0, 1, 0, 1, 0, 0, 1, 0): 1000,
        (0, 1, 0, 1, 1, 0, 0, 0): 1000,
        (0, 1, 0, 1, 1, 0, 1, 0): 1000,
        (0, 1, 1, 0, 0, 0, 0, 0): 1000,
        (0, 1, 1, 0, 0, 0, 1, 0): 1000,
        (0, 1, 1, 0, 1, 0, 0, 0): 1000,
        (0, 1, 1, 0, 1, 0, 1, 0): 1000,
        (1, 0, 0, 0, 0, 0, 1, 0): 1000,
        (1, 0, 0, 0, 1, 0, 0, 0): 1000,
        (1, 0, 0, 0, 1, 0, 1, 0): 1000,
        (1, 0, 0, 1, 0, 0, 0, 0): 1000,
        (1, 0, 0, 1, 0, 0, 1, 0): 1000,
        (1, 0, 0, 1, 1, 0, 0, 0): 1000,
        (1, 0, 0, 1, 1, 0, 1, 0): 1000,
        (1, 0, 1, 0, 0, 0, 0, 0): 1000,
        (1, 0, 1, 0, 0, 0, 1, 0): 2000, # 這是 2000 筆
        (1, 0, 1, 0, 1, 0, 0, 0): 1000,
        (1, 0, 1, 0, 1, 0, 1, 0): 1000,
    }

    # --- 載入單一缺陷資料 ---
    data = np.load(SINGLE_DEFECT_FILE)
    X_single = data['arr_0']
    y_single = data['arr_1']

    single_defect_pools = {i: [] for i in range(8)}
    for img, label in zip(X_single, y_single):
        heatmap = img.astype(np.float32)
        if heatmap.max() > 0:
            heatmap = heatmap / (heatmap.max() + 1e-9)
        binary_map = (heatmap > 0.1).astype(np.uint8)
        clean_mask, _ = extract_roi(binary_map)
        defect_index = np.argmax(label)
        single_defect_pools[defect_index].append((heatmap, clean_mask))

    # --- 合成 ---
    X_synthetic, y_synthetic = [], []
    for target_label_tuple, total_quantity in target_synthesis_list.items():
        target_label = np.array(target_label_tuple, dtype=np.float32)
        defect_indices = np.where(target_label==1)[0]
        if any(len(single_defect_pools[idx])==0 for idx in defect_indices):
            print(f"資料池不足，跳過 {target_label_tuple}")
            continue
        for _ in tqdm(range(total_quantity)):
            images_to_merge = [random.choice(single_defect_pools[idx]) for idx in defect_indices]
            merged = merge_image_list(images_to_merge)
            if merged is not None:
                # --- 實作公式 (9): 自我調整閾值 ---
                # 論文參數 lambda = 0.5 
                lambda_thresh = 0.5
                mu_h = np.mean(merged)
                sigma_h = np.std(merged)
                tau_adaptive = mu_h + lambda_thresh * sigma_h
                
                # --- 實作公式 (10): 量化 ---
                # 產生二值化圖像
                binary_mixed_map = (merged > tau_adaptive).astype(np.int32) # 轉為 0 和 1

                X_synthetic.append(binary_mixed_map)
                y_synthetic.append(target_label.astype(np.int32))

    # --- 儲存 npz ---
    if X_synthetic:
        X_synthetic_arr = np.array(X_synthetic, dtype=np.int32)
        y_synthetic_arr = np.array(y_synthetic, dtype=np.int32)
        np.savez_compressed(
            SYNTHETIC_OUTPUT_FILE,
            arr_0=X_synthetic_arr,
            arr_1=y_synthetic_arr
        )
        print(f"合成完成: {X_synthetic_arr.shape}, {y_synthetic_arr.shape}")


