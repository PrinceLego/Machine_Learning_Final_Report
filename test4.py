import numpy as np

print("--- 正在檢查 `wafer_Map_Datasets.npz` 的 `arr_0` 內容 ---")

try:
    # 確保使用您 .npz 檔案的正確路徑
    data = np.load('MixedWM38_modified.npz')
    X_raw = data['arr_0']
    
    print(f"X (arr_0) 的資料型態: {X_raw.dtype}")
    print(f"X (arr_0) 的形狀: {X_raw.shape}")
    
    # 檢查最大值和最小值
    min_val = np.min(X_raw)
    max_val = np.max(X_raw)
    print(f"X (arr_0) 的最小值: {min_val}")
    print(f"X (arr_0) 的最大值: {max_val}")
    
    # 檢查獨特值 (如果獨特值不多，就全部印出)
    unique_vals = np.unique(X_raw)
    
    if len(unique_vals) < 30:
        print(f"X (arr_0) 的所有獨特值: {unique_vals}")
    else:
        # 如果獨特值太多 (像影像資料)，就只印出前幾個
        print(f"X (arr_0) 有 {len(unique_vals)} 個獨特值 (範例: {unique_vals[:10]}...)")

except FileNotFoundError:
    print("❌ 錯誤：找不到 .npz 檔案。請檢查路徑。")
except Exception as e:
    print(f"❌ 發生錯誤: {e}")

print("--- 檢查完畢 ---")