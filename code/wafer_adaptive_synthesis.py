import os
import numpy as np
import cv2
from skimage import measure, morphology
import random
from tqdm import tqdm
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import linalg
import matplotlib.pyplot as plt

def extract_roi(binary_map, area_min=25, dilate_radius=1, erode_radius=1):
    # ... (原本的程式碼) ...
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

def merge_image_list(images_and_weights_to_merge, max_rotation=360, flip_prob=0.5, max_translate=4):

    if not images_and_weights_to_merge:
        return None
    
    h, w = images_and_weights_to_merge[0][0].shape
    H_sum = np.zeros((h, w), dtype=np.float32)
    INTERPOLATION_METHOD = cv2.INTER_NEAREST 
    BORDER_METHOD = cv2.BORDER_CONSTANT
    BORDER_VALUE = 0.0 

    for heatmap, weight_map in images_and_weights_to_merge:
        angle = random.uniform(0, max_rotation)
        M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        
        do_flip_lr = False
        do_flip_ud = False
        if random.random() < flip_prob:
            if random.random() < 0.5:
                do_flip_lr = True 
            else:
                do_flip_ud = True 
        
        tx = random.randint(-max_translate, max_translate)
        ty = random.randint(-max_translate, max_translate)
        M_trans = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)

        def apply_same_augmentation(img):
            out = cv2.warpAffine(img, M_rot, (w, h), 
                                 flags=INTERPOLATION_METHOD, 
                                 borderMode=BORDER_METHOD, 
                                 borderValue=BORDER_VALUE)
            if do_flip_lr:
                out = np.fliplr(out)
            if do_flip_ud:
                out = np.flipud(out)
            return out

        H_aug = apply_same_augmentation(heatmap)
        W_aug = apply_same_augmentation(weight_map)
        H_sum += W_aug * H_aug 

    merged = H_sum
    if merged.max() > 0:
        merged = merged / (merged.max() + 1e-9)

    global wafer_mask
    merged *= wafer_mask

    return merged.astype(np.float32)

def calculate_fid(real_features, syn_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = syn_features.mean(axis=0), np.cov(syn_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def get_inception_features(images, model):
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
    images_rgb = np.repeat(images, 3, axis=-1)
    images_resized = tf.image.resize(images_rgb, [299, 299])
    images_preprocessed = tf.keras.applications.inception_v3.preprocess_input(images_resized)
    features = model.predict(images_preprocessed, verbose=0)
    return features

def evaluate_synthesis_quality(synthetic_file, real_file):

    print(f"\n--- 開始定量評估 ---")
    print(f"合成資料: {synthetic_file}")
    print(f"真實資料: {real_file}")

    # 1. 載入資料
    try:
        syn_data = np.load(synthetic_file)
        real_data = np.load(real_file)
        
        X_syn = syn_data['arr_0']
        y_syn = syn_data['arr_1']
        
        X_real = real_data['arr_0']
        y_real = real_data['arr_1']
        
        # 簡單檢查：如果兩個檔案的資料完全一樣，FID 就會是 0
        if np.array_equal(X_syn, X_real):
            print("\n警告：合成資料與真實資料完全相同！")
            print("   請檢查 REAL_MIXED_DEFECT_FILE 路徑是否正確，")
            print("   或者您是否不小心將真實資料覆蓋掉了。")

    except FileNotFoundError:
        print("錯誤：找不到資料檔案，無法進行評估。")
        return

    print("正在載入 InceptionV3 模型 (用於 FID)...")
    # 關閉 verbose 避免雜訊
    inception_model = tf.keras.applications.inception_v3.InceptionV3(
        include_top=False, pooling='avg', input_shape=(299, 299, 3))

    def to_visual(img):
        # 轉為 0-255 視覺化格式
        vis = np.zeros_like(img, dtype=np.float32)
        vis[img==1] = 128
        vis[img==2] = 255
        return vis

    # --- 2. 依照類別分組計算 SSIM / PSNR ---
    print("正在依照「相同缺陷類別」配對並計算 SSIM/PSNR...")
    
    ssim_scores = []
    psnr_scores = []
    
    # 找出所有出現過的 unique labels (轉為 tuple 才能當 dict key)
    # y_syn 是 one-hot 或 multi-hot 陣列
    syn_labels = [tuple(row) for row in y_syn]
    real_labels = [tuple(row) for row in y_real]
    
    unique_labels = set(syn_labels).intersection(set(real_labels))
    
    if len(unique_labels) == 0:
        print("錯誤：合成資料與真實資料沒有共同的缺陷類別，無法配對比較！")
        return

    for label in tqdm(unique_labels):
        # 找出該類別的所有索引
        idxs_s = [i for i, x in enumerate(syn_labels) if x == label]
        idxs_r = [i for i, x in enumerate(real_labels) if x == label]
        
        # 該類別取樣數量 (取兩者最小值，最多測 50 張以節省時間)
        n_samples = min(len(idxs_s), len(idxs_r), 50)
        
        if n_samples == 0: continue
        
        # 隨機選出該類別的樣本
        sel_s = np.random.choice(idxs_s, n_samples, replace=False)
        sel_r = np.random.choice(idxs_r, n_samples, replace=False)
        
        for i in range(n_samples):
            im_s = to_visual(X_syn[sel_s[i]])
            im_r = to_visual(X_real[sel_r[i]])
            
            s = ssim(im_s, im_r, data_range=255)
            p = psnr(im_r, im_s, data_range=255)
            
            ssim_scores.append(s)
            psnr_scores.append(p)

    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
    avg_psnr = np.mean(psnr_scores) if psnr_scores else 0.0

    # --- 3. 計算 FID (整體隨機取樣) ---
    print("正在提取特徵以計算 FID...")
    
    # 為了 FID，我們隨機取樣整體分佈 (例如各取 1000 張)
    fid_sample_size = min(len(X_syn), len(X_real), 1000)
    idx_syn = np.random.choice(len(X_syn), fid_sample_size, replace=False)
    idx_real = np.random.choice(len(X_real), fid_sample_size, replace=False)
    
    feats_syn = get_inception_features(to_visual(X_syn[idx_syn]), inception_model)
    feats_real = get_inception_features(to_visual(X_real[idx_real]), inception_model)
    
    fid_score = calculate_fid(feats_real, feats_syn)

    # --- 輸出結果 ---
    print("\n" + "="*30)
    print("合成資料品質評估結果 (修正版)")
    print("="*30)
    print(f"SSIM (結構相似性) : {avg_ssim:.4f} (目標 > 0.7)")
    print(f"PSNR (峰值信噪比) : {avg_psnr:.2f} dB (目標 > 20 dB)")
    print(f"FID  (特徵距離)   : {fid_score:.2f} (越低越好，非 0)")
    print("="*30)


if __name__ == "__main__":
    SINGLE_DEFECT_FILE = 'single_defect_wafers.npz'
    SYNTHETIC_OUTPUT_FILE = 'synthetic_mixed_dataset.npz'
    NORMAL_DEFECT_FILE = 'normal_wafers.npz'

    REAL_MIXED_DEFECT_FILE = 'multi_defect_wafers.npz' 

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
        (1, 0, 1, 0, 0, 0, 1, 0): 2000,  # 這是 2000 筆
        (1, 0, 1, 0, 1, 0, 0, 0): 1000,
        (1, 0, 1, 0, 1, 0, 1, 0): 1000,
    }

    # 1. 載入資料
    data = np.load(SINGLE_DEFECT_FILE)
    X_single = data['arr_0']
    y_single = data['arr_1']
    wafer_mask = (X_single[0] > 0).astype(np.uint8)

    try:
        normal_data = np.load(NORMAL_DEFECT_FILE)
        normal_wafer_pool = normal_data['arr_0'].astype(np.int32)
        print(f"成功載入 {len(normal_wafer_pool)} 筆背景雜訊來源。")
    except:
        print("警告：無法載入背景雜訊，將使用純淨背景。")
        normal_wafer_pool = []

    # 2. 預處理
    print("Pre-calculating heatmaps and weight maps...")
    single_defect_pools = {i: [] for i in range(8)}
    for img, label in tqdm(zip(X_single, y_single)):
        if np.sum(label) != 1: continue
        
        heatmap = img.astype(np.float32)
        heatmap[heatmap == 1] = 0.5
        heatmap[heatmap == 2] = 1.0
        
        binary_map = (img == 2).astype(np.uint8)
        clean_mask, (xc, yc) = extract_roi(binary_map) 
        weight_map = compute_weight_map(heatmap, clean_mask, (xc, yc), alpha=1.0, beta=1.0, sigma=26.0)
        
        defect_index = np.argmax(label)
        single_defect_pools[defect_index].append((heatmap, weight_map))


    # --- 合成 (*** 量化區塊已修改 ***) ---
    X_synthetic, y_synthetic = [], []
    for target_label_tuple, total_quantity in target_synthesis_list.items():
        target_label = np.array(target_label_tuple, dtype=np.float32)
        defect_indices = np.where(target_label==1)[0]
        if any(len(single_defect_pools[idx])==0 for idx in defect_indices):
            print(f"資料池不足，跳過 {target_label_tuple}")
            continue
        
        print(f"正在合成 {total_quantity} 筆 {target_label_tuple}...")
        for _ in tqdm(range(total_quantity)):

            images_to_merge = [random.choice(single_defect_pools[idx]) for idx in defect_indices]
            

            merged = merge_image_list(images_to_merge) 
            
            if merged is not None:
                lambda_thresh = 0.5
                valid_area = wafer_mask > 0
                if np.any(merged[valid_area]):
                    mu_h = np.mean(merged[valid_area])
                    sigma_h = np.std(merged[valid_area])
                else:
                    mu_h = 0.0
                    sigma_h = 0.0
                tau_adaptive = mu_h + lambda_thresh * sigma_h


                

                final_map = wafer_mask.copy().astype(np.int32)


                defect_mask = (merged > tau_adaptive) & (wafer_mask > 0)
                
                final_map[defect_mask] = 2
                
  
                # ------------------------------------------------------
        
                if len(normal_wafer_pool) > 0: 

                    noise_source_img = random.choice(normal_wafer_pool)

                    noise_mask = (noise_source_img == 2) & (wafer_mask > 0)
                    

                    final_map[noise_mask] = 2
                # ------------------------------------------------------


                

                X_synthetic.append(final_map)
                y_synthetic.append(target_label.astype(np.int32))

    if X_synthetic:
        X_synthetic_arr = np.array(X_synthetic, dtype=np.int32)
        y_synthetic_arr = np.array(y_synthetic, dtype=np.int32)
        np.savez_compressed(
            SYNTHETIC_OUTPUT_FILE,
            arr_0=X_synthetic_arr,
            arr_1=y_synthetic_arr
        )
        print(f"合成完成: {X_synthetic_arr.shape}, {y_synthetic_arr.shape}")
    else:
        print("沒有合成任何影像。")


    if os.path.exists(REAL_MIXED_DEFECT_FILE):
         evaluate_synthesis_quality(SYNTHETIC_OUTPUT_FILE, REAL_MIXED_DEFECT_FILE)
    else:
         print(f"提示: 未找到真實混合缺陷資料 '{REAL_MIXED_DEFECT_FILE}'，跳過定量評估。")




def save_comparison_image(syn_file, real_file, output_img="comparison_result.png"):
    try:
        syn_data = np.load(syn_file)['arr_0']
        real_data = np.load(real_file)['arr_0']
        syn_y = np.load(syn_file)['arr_1']
        real_y = np.load(real_file)['arr_1']
    except:
        return

    indices = np.random.choice(len(syn_data), 5, replace=False)
    
    fig, axes = plt.subplots(5, 2, figsize=(8, 20))
    fig.suptitle("Left: Synthetic | Right: Real (Random Sample)", fontsize=16)

    for i, idx in enumerate(indices):
        current_label = syn_y[idx]
        
        axes[i, 0].imshow(syn_data[idx], cmap='inferno')
        axes[i, 0].set_title(f"Syn: {current_label}")
        axes[i, 0].axis('off')


        matching_real_indices = [k for k, y in enumerate(real_y) if np.array_equal(y, current_label)]
        
        if matching_real_indices:
            real_idx = random.choice(matching_real_indices)
            axes[i, 1].imshow(real_data[real_idx], cmap='inferno')
            axes[i, 1].set_title(f"Real: {real_y[real_idx]}")
        else:
            axes[i, 1].text(0.5, 0.5, "No Real Match found", ha='center')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_img)
    print(f"視覺化對比圖已儲存至: {output_img} (請打開檢查)")

if os.path.exists(REAL_MIXED_DEFECT_FILE):
    save_comparison_image(SYNTHETIC_OUTPUT_FILE, REAL_MIXED_DEFECT_FILE)