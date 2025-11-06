import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import datetime
import time
import os
import matplotlib.pyplot as plt

# === 1️⃣ GPU 偵測 ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU 已啟用：", gpus)
else:
    print("⚠️ 未偵測到 GPU，將使用 CPU")

# === 2️⃣ 載入資料 ===
try:
    data = np.load('/Users/prince_lego/Documents/program/Database/wafer_Map_Datasets.npz')
    X = data['arr_0']
    y = data['arr_1']
except FileNotFoundError:
    print("❌ 錯誤：找不到 .npz 檔案。請檢查路徑是否正確。")
    exit() 

print("原始 X shape:", X.shape)
print("原始 y shape:", y.shape)

# === 3️⃣ 正規化影像 ===
X = X.astype('float32') / 255.0
X = np.expand_dims(X, axis=-1)  # (N, 52, 52, 1)

# === 4️⃣ 分割 訓練 / 驗證 / 測試集 ===
# 第一次分割：20% 測試集
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42
)

# 第二次分割：剩下 80% 分為 60% 訓練 + 20% 驗證
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, 
    test_size=0.25, # 80% * 0.25 = 20%
    random_state=42
)

# +++ Y 標籤檢查 (使用訓練集) +++
print("--- Y 標籤檢查 (Train) ---")
print("y_train 的前 5 筆資料:\n", y_train[:5])
print("y_train 的獨特值:", np.unique(y_train))
print("--------------------------")

print(f"訓練集大小 (Train):   {X_train.shape} (佔 60%)")
print(f"驗證集大小 (Val):     {X_val.shape} (佔 20%)")
print(f"測試集大小 (Test):    {X_test.shape} (佔 20%)")


# === 5️⃣ 建立 CNN 模型 (multi-label) ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(52, 52, 1)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(8, activation='sigmoid')  # multi-label
])

model.summary()

# === 6️⃣ (修改) 編譯模型 (multi-label) ===
# 降低學習率
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
        # --- 這裡修改了 ---
        tf.keras.metrics.BinaryAccuracy(name='accuracy'), # 將 'binary_acc' 改名為 'accuracy'
        tf.keras.metrics.AUC(name='auc')
        # --- 修改結束 ---
    ]
)

# === 7️⃣ TensorBoard callback ===
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# === 8️⃣ EarlyStopping & ModelCheckpoint ===
callbacks = [
    tensorboard_cb,
    #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_wafer_cnn_model.keras', save_best_only=True, monitor='val_auc', mode='max') 
]

# === 9️⃣ 訓練模型 ===
start_time = time.time()
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)
end_time = time.time()
print(f"訓練總時間：{end_time - start_time:.2f} 秒")

# === 1️⃣0️⃣ 儲存最終模型 ===
model.save('wafer_cnn_model_final.keras')

"""
# === 1️⃣1️⃣ (修改) 繪製訓練結果 ===
plt.figure(figsize=(16, 6))



# --- 這裡修改了 ---
# Accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='train accuracy') # 'binary_acc' -> 'accuracy'
plt.plot(history.history['val_accuracy'], label='val accuracy') # 'val_binary_acc' -> 'val_accuracy'
plt.title('Accuracy') # 'Binary Accuracy' -> 'Accuracy'
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
# --- 修改結束 ---

# Loss
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# AUC
plt.subplot(1, 3, 3)
plt.plot(history.history['auc'], label='train auc')
plt.plot(history.history['val_auc'], label='val auc')
plt.title('AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
"""


# === 1️⃣2️⃣ (修改) 在「測試集」上評估最佳模型 ===
print("\n" + "="*30)
print(" 正在載入「最佳模型」進行最終評估... ")
print(" (模型儲存於 'best_wafer_cnn_model.keras')")
print("="*30)

try:
    best_model = tf.keras.models.load_model('best_wafer_cnn_model.keras')
except Exception as e:
    print(f"❌ 載入 'best_wafer_cnn_model.keras' 失敗: {e}")
    print("--- 將使用訓練完成的「最終模型」進行評估 (可能非最佳) ---")
    best_model = model

print("\n--- 正在評估「測試集」(Test Set) ---")
test_results = best_model.evaluate(X_test, y_test, verbose=1)

print("\n--- 🚀 最終測試結果 (Test Set) 🚀 ---")
# test_results 列表的順序與 model.compile 中的 metrics 相同
print(f"  測試集 Loss:            {test_results[0]:.4f}")
# --- 這裡修改了 ---
print(f"  測試集 Accuracy:        {test_results[1]:.4f}") # 'Binary Accuracy' -> 'Accuracy'
# --- 修改結束 ---
print(f"  測試集 AUC:             {test_results[2]:.4f}")
print("="*38)