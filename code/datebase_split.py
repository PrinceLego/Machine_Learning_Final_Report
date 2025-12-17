import numpy as np
import os


INPUT_FILE_PATH = '/Users/prince_lego/Documents/program/Database/MixedWM38/wafer_Map_Datasets.npz' #pro


NORMAL_FILE = 'normal_wafers.npz'
SINGLE_FILE = 'single_defect_wafers.npz'
MULTI_FILE = 'multi_defect_wafers.npz'

print(f"正在從: {INPUT_FILE_PATH} 載入資料...")

# --- 2. 載入資料 ---
try:
    data = np.load(INPUT_FILE_PATH)
    arr_0 = data['arr_0']
    arr_1 = data['arr_1']
    print(f"資料載入成功。 X shape: {arr_0.shape}, y shape: {arr_1.shape}")
except FileNotFoundError:
    print(f"錯誤：在 '{INPUT_FILE_PATH}' 找不到檔案。")
    print("請檢查 INPUT_FILE_PATH 變數是否設定正確。")
    exit()
except Exception as e:
    print(f"載入檔案時發生錯誤: {e}")
    exit()


label_sums = np.sum(arr_1, axis=1) 


normal_indices = np.where(label_sums == 0)[0]
single_indices = np.where(label_sums == 1)[0]
multi_indices = np.where(label_sums > 1)[0]

print(f"{len(normal_indices)} 筆正常, {len(single_indices)} 筆單一缺陷, {len(multi_indices)} 筆多重缺陷。")


def process_and_save(indices, category_name, filename):

    if len(indices) == 0:
        print(f"\n類別 {category_name} 中沒有資料。")
        return

    print("\n" + "="*50)
    print(f"處理中: {category_name}")
    print(f"輸出檔案: {filename}")

    # 根據索引提取資料
    X_subset = arr_0[indices]
    y_subset = arr_1[indices]
    
    # 儲存為新的 .npz 檔案
    np.savez_compressed(
        filename,
        arr_0=X_subset,
        arr_1=y_subset
    )
    print(f"成功儲存 {len(indices)} 筆資料到 {filename}.")

    # --- 產生您需要的詳細索引報告 ---
    print(f"\n--- {category_name} ( {filename} ) 詳細索引報告 ---")
    print(f"總數量: {len(indices)}")
    print(f"新檔案索引範圍: 0 到 {len(indices) - 1}")
    print(f"原始索引範圍 (在 wafer_Map_Datasets.npz 中): 從 {indices[0]} 到 {indices[-1]}")
    
    print("\n標籤細節 (Label Breakdown):")
    
    # 找出所有不重複的標籤 (unique labels)
    unique_labels = np.unique(y_subset, axis=0)
    
    for label in unique_labels:
        # 1. 找到這個標籤在 'y_subset' 中的索引 (新檔案中的索引)
        subset_indices = np.where((y_subset == label).all(axis=1))[0]
        
        # 2. 透過 'indices' 陣列，將它們映射回 'wafer_Map_Datasets.npz' 的原始索引
        original_indices_for_label = indices[subset_indices]
        
        count = len(original_indices_for_label)
        
        print(f"\n  --- 標籤 {str(label)} ---")
        print(f"    - 總數量: {count}")
        
        # --- 這是您要求的修改 ---
        # 僅顯示該標籤的第一個索引和最後一個索引
        if count > 0:
            start_index = original_indices_for_label[0]
            end_index = original_indices_for_label[-1]
            print(f"    - 原始索引範圍: {start_index}-{end_index}")
        else:
            print(f"    - 原始索引範圍: N/A")
    
    print("="*50)

# --- 5. 執行所有類別的處理 ---
process_and_save(normal_indices, "正常 (Normal)", NORMAL_FILE)
process_and_save(single_indices, "單一缺陷 (Single-Defect)", SINGLE_FILE)
process_and_save(multi_indices, "多重缺陷 (Multi-Defect)", MULTI_FILE)

print("\n腳本執行完畢。三個 .npz 檔案已儲存。")