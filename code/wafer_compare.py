import numpy as np
import matplotlib.pyplot as plt
import os
import random


FILE_PATHS = [
    'synthetic_mixed_dataset_noise.npz',  
    'synthetic_mixed_dataset_nonoise.npz',      
    'multi_defect_wafers.npz'       
]


DATASET_LABELS = ['Add noise automatically', 'Based on the paper', 'MixedWM38']


OUTPUT_DIR = 'all_classes_comparison'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DEFECT_NAMES = ['C', 'D', 'EL', 'ER', 'L', 'NF', 'S', 'R']
CLASS_TUPLES = [
    (0, 0, 0, 0, 1, 0, 1, 0), (0, 0, 0, 1, 0, 0, 1, 0), (0, 0, 0, 1, 1, 0, 0, 0),
    (0, 0, 0, 1, 1, 0, 1, 0), (0, 0, 1, 0, 0, 0, 1, 0), (0, 0, 1, 0, 1, 0, 0, 0),
    (0, 0, 1, 0, 1, 0, 1, 0), (0, 1, 0, 0, 0, 0, 1, 0), (0, 1, 0, 0, 1, 0, 0, 0),
    (0, 1, 0, 0, 1, 0, 1, 0), (0, 1, 0, 1, 0, 0, 0, 0), (0, 1, 0, 1, 0, 0, 1, 0),
    (0, 1, 0, 1, 1, 0, 0, 0), (0, 1, 0, 1, 1, 0, 1, 0), (0, 1, 1, 0, 0, 0, 0, 0),
    (0, 1, 1, 0, 0, 0, 1, 0), (0, 1, 1, 0, 1, 0, 0, 0), (0, 1, 1, 0, 1, 0, 1, 0),
    (1, 0, 0, 0, 0, 0, 1, 0), (1, 0, 0, 0, 1, 0, 0, 0), (1, 0, 0, 0, 1, 0, 1, 0),
    (1, 0, 0, 1, 0, 0, 0, 0), (1, 0, 0, 1, 0, 0, 1, 0), (1, 0, 0, 1, 1, 0, 0, 0),
    (1, 0, 0, 1, 1, 0, 1, 0), (1, 0, 1, 0, 0, 0, 0, 0), (1, 0, 1, 0, 0, 0, 1, 0),
    (1, 0, 1, 0, 1, 0, 0, 0), (1, 0, 1, 0, 1, 0, 1, 0)
]

def tuple_to_string(t):
    names = [DEFECT_NAMES[i] for i, v in enumerate(t) if v == 1]
    return "+".join(names)

CLASS_NAMES = [tuple_to_string(t) for t in CLASS_TUPLES]
LABEL_MAP = {tuple(t): i for i, t in enumerate(CLASS_TUPLES)}

def convert_labels_to_int(y_batch_multi_label):
    y_batch_int = []
    for y_vec in y_batch_multi_label:
        y_tuple = tuple(y_vec)
        if y_tuple in LABEL_MAP:
            y_batch_int.append(LABEL_MAP[y_tuple])
        else:
            y_batch_int.append(-1)
    return np.array(y_batch_int)

# --- 3. 載入資料函式 ---
def load_single_dataset(file_path, label_name):
    print(f"正在載入 {label_name}: {file_path}")
    if not os.path.exists(file_path):
        print(f"錯誤: 找不到檔案 {file_path}")
        return None, None

    try:
        data = np.load(file_path)
        X = data['arr_0']
        y_raw = data['arr_1']
        y_int = convert_labels_to_int(y_raw)
        return X, y_int
    except Exception as e:
        print(f"載入失敗 {file_path}: {e}")
        return None, None

def plot_and_save_comparison(selected_images, class_name, dataset_labels, class_idx):
    n_datasets = len(selected_images)
    fig, axes = plt.subplots(nrows=1, ncols=n_datasets, figsize=(5 * n_datasets, 5))
    

    fig.suptitle(f"{class_name}", fontsize=16)

    for i in range(n_datasets):
        ax = axes[i] if n_datasets > 1 else axes
        img = selected_images[i]
        label = dataset_labels[i]

        if img is not None:
            im = ax.imshow(img, cmap='viridis', vmin=0, vmax=2)
            ax.set_title(label)
        else:

            ax.text(0.5, 0.5, 'No Sample', horizontalalignment='center', 
                    verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f"{label} (Missing)")
        
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    

    filename = f"{class_idx:02d}_{class_name}.png"

    
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    
    plt.close(fig) 

def main():
    print("=== 開始載入資料集 ===")
    datasets = [] 
    valid_data_count = 0

    for path, label in zip(FILE_PATHS, DATASET_LABELS):
        X, y = load_single_dataset(path, label)
        datasets.append((X, y))
        if X is not None:
            valid_data_count += 1
    
    if valid_data_count == 0:
        print("沒有成功載入任何資料集，程式結束。")
        return

    print(f"\n=== 開始批次處理 {len(CLASS_NAMES)} 個類別 ===")
    print(f"圖片將儲存至資料夾: {OUTPUT_DIR}/")
    print("-" * 50)

    for target_index, target_name in enumerate(CLASS_NAMES):
        
        selected_images = []
        found_any = False 


        for idx, (X, y) in enumerate(datasets):
            if X is None:
                selected_images.append(None)
                continue


            indices = np.where(y == target_index)[0]
            
            if len(indices) > 0:

                random_idx = np.random.choice(indices)
                img = X[random_idx]
                selected_images.append(img)
                found_any = True
            else:
                selected_images.append(None)

        if found_any:
            plot_and_save_comparison(selected_images, target_name, DATASET_LABELS, target_index)
            print(f"[{target_index+1}/{len(CLASS_NAMES)}] 已儲存: {target_name}")
        else:
            print(f"[{target_index+1}/{len(CLASS_NAMES)}] 跳過: {target_name} (所有資料集皆無此類別數據)")

    print("\n=== 所有作業完成 ===")

if __name__ == "__main__":
    main()