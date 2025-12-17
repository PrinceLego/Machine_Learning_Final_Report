import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import datetime
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns


current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

base_dir = "experiments_multilabel"

RUN_DIR = os.path.join(base_dir, current_timestamp)


if not os.path.exists(RUN_DIR):
    os.makedirs(RUN_DIR)

BEST_MODEL_PATH = os.path.join(RUN_DIR, 'best_wafer_SACNN_multilabel.keras')
FINAL_MODEL_PATH = os.path.join(RUN_DIR, 'final_wafer_SACNN_multilabel.keras')
LOG_FILE_PATH = os.path.join(RUN_DIR, 'execution_result.txt')
HISTORY_IMG_PATH = os.path.join(RUN_DIR, 'training_history_multilabel.png')
CM_IMG_PATH = os.path.join(RUN_DIR, 'confusion_matrix_multilabel.png')
TENSORBOARD_LOG_DIR = os.path.join(RUN_DIR, 'logs')


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU 已啟用：", gpus)
else:
    print("未偵測到 GPU，將使用 CPU")

TRAIN_FILE_PATH = 'synthetic_mixed_dataset_noise.npz'
TEST_FILE_PATH = 'multi_defect_wafers.npz'

try:
    print(f"正在載入訓練資料... {TRAIN_FILE_PATH}")
    train_data = np.load(TRAIN_FILE_PATH)
    X_train_full = train_data['arr_0']
    y_train_full = train_data['arr_1']
    
    print(f"正在載入測試資料... {TEST_FILE_PATH}")
    test_data = np.load(TEST_FILE_PATH)
    X_test = test_data['arr_0']
    y_test = test_data['arr_1']
    
except Exception as e:
    print(f"錯誤：資料載入失敗。 {e}")
    exit()


X_train_full = X_train_full.astype('float32') / 2.0
X_train_full = np.expand_dims(X_train_full, axis=-1)
X_test = X_test.astype('float32') / 2.0
X_test = np.expand_dims(X_test, axis=-1)


X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, 
    test_size=0.2, 
    random_state=42
)


BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = len(X_train)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("tf.data 管線建立完成。")


def self_attention_layer(x, block_id):
    attention_scores = tf.keras.layers.Conv2D(1, (1, 1), activation='linear', name=f'sa_conv_block{block_id}')(x)
    attention_weights = tf.keras.layers.Softmax(axis=(1, 2), name=f'sa_softmax_block{block_id}')(attention_scores)
    scaled_features = tf.keras.layers.multiply([x, attention_weights], name=f'sa_multiply_block{block_id}')
    return scaled_features

def build_sacnn_model(input_shape=(52, 52, 1), num_classes=8):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # --- Block 1 ---
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='valid')(inputs) 
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    # --- Block 2 (Self-Attention) ---
    x = self_attention_layer(x, block_id=1) 

    # --- Block 3 ---
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='valid')(x) 
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    # --- Block 4 (Self-Attention) ---
    x = self_attention_layer(x, block_id=2)

    # --- Block 5 ---
    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    # --- Classifier Head ---
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # --- 輸出層 (Sigmoid for Multi-label) ---
    outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

num_classes = y_train.shape[1]
model = build_sacnn_model(input_shape=(52, 52, 1), num_classes=num_classes)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='auc')
    ]
)


tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOG_DIR, histogram_freq=1)

callbacks = [
    tensorboard_cb,
    tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, monitor='val_auc', mode='max') 
]


start_time = time.time()
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    callbacks=callbacks,
    verbose=1
)
end_time = time.time()
print(f"訓練總時間：{end_time - start_time:.2f} 秒")


model.save(FINAL_MODEL_PATH)
print(f"最終模型已儲存至: {FINAL_MODEL_PATH}")

print("\n" + "="*30)
print(" 正在評估最佳模型... ")
print("="*30)
try:
    best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
except:
    print("載入最佳模型失敗，使用當前模型")
    best_model = model

test_results = best_model.evaluate(test_dataset, verbose=1)
print(f"測試集 Loss: {test_results[0]:.4f}, Accuracy: {test_results[1]:.4f}, AUC: {test_results[2]:.4f}")



print("\n正在繪製訓練歷史...")
plt.figure(figsize=(20, 8))

# 準確率圖
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy (Binary Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss 圖
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss (Binary Crossentropy)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(HISTORY_IMG_PATH)
print(f" 訓練歷史圖已儲存為 {HISTORY_IMG_PATH}")

# 繪製混淆矩陣
print("\n正在計算並繪製混淆矩陣...")

# 預測測試集
y_pred_probs = best_model.predict(X_test)
y_pred_binary = (y_pred_probs > 0.5).astype(int)

# 原始所有標籤名稱
ALL_DEFECT_NAMES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-Full', 'Scratch', 'Random']

mcm = multilabel_confusion_matrix(y_test, y_pred_binary)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for i, (matrix, name) in enumerate(zip(mcm, ALL_DEFECT_NAMES)):
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Not ' + name, name],
                yticklabels=['Not ' + name, name])
    axes[i].set_title(f'{name} Confusion Matrix')
    axes[i].set_ylabel('True Label')
    axes[i].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(CM_IMG_PATH)
print(f"多標籤混淆矩陣已儲存為 {CM_IMG_PATH}")

active_indices = [0, 1, 2, 3, 4, 6]
active_names = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Scratch']

# 生成過濾後的分類報告
class_report_str = classification_report(
    y_test, 
    y_pred_binary, 
    labels=active_indices, 
    target_names=active_names,
    zero_division=0
)

log_content = (
    f"[{current_timestamp}] 實驗執行報告 (Multi-Label)\n"
    f"{'='*50}\n"
    f"  - 實驗目錄:   {RUN_DIR}\n"
    f"  - 資料集:     {TEST_FILE_PATH}\n"
    f"  - 模型架構:   SACNN (8 Classes, Sigmoid)\n"
    f"  - Epochs:     {len(history.history['loss'])}\n"
    f"  - 訓練總時間: {end_time - start_time:.2f} 秒\n"
    f"{'-'*30}\n"
    f"  ★ Test Loss:     {test_results[0]:.4f}\n"
    f"  ★ Test Accuracy: {test_results[1]:.4f}\n"
    f"  ★ Test AUC:      {test_results[2]:.4f}\n"
    f"{'='*50}\n"
    f"詳細分類報告 (已排除無樣本的 NF 與 R):\n"
    f"{class_report_str}\n"
    f"{'='*50}\n"
)

try:
    with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(log_content)
    print(f"\n測試報告已寫入： {LOG_FILE_PATH}")
    # 顯示報告
    print("\n--- Classification Report (Filtered) ---")
    print(class_report_str)
except Exception as e:
    print(f"\n寫入紀錄檔失敗： {e}")

print(f"\n所有實驗檔案已保存於: {RUN_DIR}")