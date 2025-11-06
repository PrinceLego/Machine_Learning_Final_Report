import numpy as np
import matplotlib.pyplot as plt
import os

# 我們將假設上傳的檔案名稱為 'MixedWM38.npz'
original_file_name = '/Users/prince_lego/Downloads/MixedWM38.npz'

print(f"--- 正在處理 '{original_file_name}' ---")
print("--- 包含： 統計、轉換、並儲存新檔案 ---")

# 檢查檔案是否存在 (使用者上傳後)
if not os.path.exists(original_file_name):
    print(f"\n❌ 錯誤：找不到檔案 '{original_file_name}'。")
    print("請確認您已上傳 'MixedWM38.npz' 檔案。")
    
else:
    try:
        # 載入 .npz 檔案
        data = np.load(original_file_name)
        
        # 檢查 'arr_0' 和 'arr_1' 是否都存在
        if 'arr_0' not in data:
            print("❌ 錯誤：檔案中找不到 'arr_0'。")
        elif 'arr_1' not in data:
            print("❌ 錯誤：檔案中找不到 'arr_1'。")
        else:
            # 讀取資料
            X_raw = data['arr_0']
            Y_labels = data['arr_1']
            
            print(f"\n[資料載入成功]")
            print(f"'arr_0' (X_raw) shape: {X_raw.shape}")
            print(f"'arr_1' (Y_labels) shape: {Y_labels.shape}")

            # ----------------------------------------------------
            # 1. 像素統計 (基於 '3')
            # ----------------------------------------------------
            total_threes_pixels = np.sum(X_raw == 3)
            print(f"\n[像素統計 (基於 '3')]")
            print(f"值為 '3' 的像素點總數: {total_threes_pixels}")

            # ----------------------------------------------------
            # 2. 晶圓圖統計 (基於 '3')
            # ----------------------------------------------------
            has_threes_mask = np.any(X_raw == 3, axis=(1, 2))
            wafers_with_threes = np.sum(has_threes_mask)
            
            print(f"\n[晶圓圖統計 (基於 '3')]")
            print(f"包含 '3' 的晶圓圖數量: {wafers_with_threes}")

            # ----------------------------------------------------
            # 3. 列出 'arr_0' 有 '3' 時對應的 'arr_1' 標籤
            # ----------------------------------------------------
            print("\n[請求 1: 列出 'arr_0' 有 '3' 時對應的 'arr_1' 標籤]")
            if wafers_with_threes > 0:
                labels_for_wafers_with_threes = Y_labels[has_threes_mask]
                print(f"找到 {len(labels_for_wafers_with_threes)} 個對應的 'arr_1' 標籤。")
                
                # 找出這些標籤的唯一值和它們的數量
                unique_labels, counts = np.unique(labels_for_wafers_with_threes, return_counts=True)
                print("這些標籤的組成：")
                for label, count in zip(unique_labels, counts):
                    print(f"  標籤 '{label}': {count} 個")
                
                indices_with_threes = np.where(has_threes_mask)[0]
                print(f"這些標籤來自 'arr_0' 的索引 (例如，前10個): {indices_with_threes[:10]}...")
            else:
                print("沒有晶圓圖包含 '3'。")

            # ----------------------------------------------------
            # 4. 視覺化
            # ----------------------------------------------------
            print("\n[視覺化]")
            if wafers_with_threes > 0:
                print("正在繪製第一張包含 '3' 的晶圓圖...")
                first_wafer_index = np.where(has_threes_mask)[0][0]
                wafer_to_show = X_raw[first_wafer_index]
                
                plt.figure(figsize=(6, 5))
                img = plt.imshow(wafer_to_show, cmap='viridis', interpolation='nearest') 
                plt.title(f"Wafer Index: {first_wafer_index} (原始資料, 包含 3)")
                unique_values = np.unique(wafer_to_show)
                plt.colorbar(ticks=unique_values, label='Pixel Value')
                
                output_image_file = "wafer_with_value_3.png"
                plt.savefig(output_image_file)
                print(f"影像已儲存至 {output_image_file}")
            else:
                print("沒有包含 '3' 的晶圓圖可供視覺化。")

            # ----------------------------------------------------
            # 5. 將 'arr_0' 中所有的 '3' 轉換為 '2'
            # ----------------------------------------------------
            print("\n[請求 2: 將 'arr_0' 中所有的 '3' 轉換為 '2']")
            if total_threes_pixels > 0:
                print("正在建立 'arr_0' 的副本並執行轉換...")
                X_modified = np.copy(X_raw)
                X_modified[X_modified == 3] = 2
                
                # --- 驗證 ---
                print("驗證轉換結果：")
                new_threes_count = np.sum(X_modified == 3)
                original_twos_count = np.sum(X_raw == 2)
                new_twos_count = np.sum(X_modified == 2)
                
                print(f"  修改前 '3' 的總數: {total_threes_pixels}")
                print(f"  修改後 '3' 的總數: {new_threes_count}")
                print(f"  修改前 '2' 的總數: {original_twos_count}")
                print(f"  修改後 '2' 的總數: {new_twos_count}")
                
                if new_twos_count == (original_twos_count + total_threes_pixels) and new_threes_count == 0:
                    print("✅ 驗證成功：所有的 '3' 都已轉換為 '2'。")
                else:
                    print("⚠️ 驗證失敗。")
                    
                # ----------------------------------------------------
                # 6. (新請求) 儲存新的 NPZ 檔案
                # ----------------------------------------------------
                print("\n[請求 3: 儲存修改後的資料]")
                output_modified_file = 'MixedWM38_modified.npz'
                print(f"正在將修改後的 'arr_0' (3 -> 2) 和 'arr_1' 儲存到 '{output_modified_file}'...")
                
                # 使用 np.savez 儲存
                np.savez(output_modified_file, arr_0=X_modified, arr_1=Y_labels)
                
                print(f"✅ 成功儲存檔案： {output_modified_file}")
                print(f"   新檔案包含：")
                print(f"   - 'arr_0': shape {X_modified.shape}, 獨特值 {np.unique(X_modified)}")
                print(f"   - 'arr_1': shape {Y_labels.shape}")

            else:
                print(" 'arr_0' 中本來就沒有 '3'，無需轉換或儲存新檔案。")

    except Exception as e:
        print(f"❌ 執行時發生未預期的錯誤: {e}")

print("\n--- 所有調查完畢 ---")