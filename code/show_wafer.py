import numpy as np
import matplotlib.pyplot as plt

# 讀取資料
#data = np.load('/Users/prince_lego/Documents/program/Database/wafer_Map_Datasets.npz')
data = np.load('/Users/prince_lego/Documents/program/Wafer_Synthesis_SACNN/multi_defect_wafers.npz')
data = np.load('/Users/prince_lego/Documents/program/Database/MixedWM38/wafer_Map_Datasets.npz')


data = np.load('/Users/prince_lego/Documents/program/Wafer_Synthesis_SACNN/normal_wafers.npz')

data = np.load('/Users/prince_lego/Documents/program/Wafer_Synthesis_SACNN/synthetic_mixed_dataset.npz')
print("Available keys:", data.files)
images = data['arr_0']
labels = data['arr_1']

print(images)
# 顯示第 0 張影像
idx = 502
plt.imshow(images[idx], cmap='Blues')
plt.title(f"Label: {labels[idx]}")
plt.show()
