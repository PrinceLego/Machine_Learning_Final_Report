import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import datetime
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU 已啟用：", gpus)
else:
    print("未偵測到 GPU，將使用 CPU")


REAL_DATA_FILE_PATH = 'multi_defect_wafers.npz' 

try:
    print(f"正在載入原始資料... {REAL_DATA_FILE_PATH}")
    real_data = np.load(REAL_DATA_FILE_PATH)
 
    X_full_real = real_data['arr_0']
    y_full_real = real_data['arr_1'] 
    
except FileNotFoundError as e:
    print(f"錯誤：找不到 .npz 檔案。請檢查路徑是否正確。 {e}")
    exit() 
except KeyError as e:
    print(f"錯誤：.npz 檔案中找不到 'arr_0' 或 'arr_1'。請檢查檔案內容。 {e}")
    exit()


X_full_real = X_full_real.astype('float32') / 2.0
X_full_real = np.expand_dims(X_full_real, axis=-1)


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
NUM_CLASSES = len(CLASS_NAMES)
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


print("\n正在將 (N, 8) 標籤轉換為 (N, 1) 整數標籤...")

y_full_real_int = convert_labels_to_int(y_full_real)


train_mask = y_full_real_int != -1
X_full_real = X_full_real[train_mask]
y_full_real_int = y_full_real_int[train_mask]


print(f"\n原始資料總大小: {X_full_real.shape}")


X_train, X_temp, y_train_int, y_temp_int = train_test_split(
    X_full_real, y_full_real_int, 
    test_size=0.3, # 30% 留給 驗證 + 測試
    random_state=42,
    stratify=y_full_real_int 
)


X_val, X_test, y_val_int, y_test_int = train_test_split(
    X_temp, y_temp_int,
    test_size=0.5, # 
    random_state=42,
    stratify=y_temp_int
)

print(f"訓練集大小 (Train):   {X_train.shape} (~70%)")
print(f"驗證集大小 (Val):     {X_val.shape} (~15%)")
print(f"測試集大小 (Test):    {X_test.shape} (~15%)")


def self_attention_layer(x, block_id):
    attention_scores = tf.keras.layers.Conv2D(1, (1, 1), activation='linear', name=f'sa_conv_block{block_id}')(x)
    attention_weights = tf.keras.layers.Softmax(axis=(1, 2), name=f'sa_softmax_block{block_id}')(attention_scores)
    scaled_features = tf.keras.layers.multiply([x, attention_weights], name=f'sa_multiply_block{block_id}')
    return scaled_features

def build_sacnn_model(input_shape=(52, 52, 1), num_classes=NUM_CLASSES):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # === 5.5 (*** 新增 ***) 訓練時資料增強 (論文 ) ===
    # 這些層只會在訓練時啟動
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = tf.keras.layers.RandomRotation(0.2)(x) # 論文使用隨機旋轉
    x = tf.keras.layers.GaussianNoise(0.01)(x) # 論文使用雜訊注入
    # === 增強結束 ===
    
    # --- Block 1 ---
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='valid')(x) 
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

    # --- Classifier Head (論文架構) ---
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = build_sacnn_model(input_shape=(52, 52, 1), num_classes=NUM_CLASSES)
model.summary()


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) 

model.compile(
    optimizer=optimizer,
    
    loss='sparse_categorical_crossentropy', 
    
    metrics=[
        # 配合 'sparse_categorical_crossentropy'，
        # 您原本使用的 'SparseCategoricalAccuracy' 也是【正確】的
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy') 
    ]
)


log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=5, 
    min_lr=1e-7,
    verbose=1
)

callbacks = [
    tensorboard_cb,
    reduce_lr_cb, 
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=12, restore_best_weights=True), 
    tf.keras.callbacks.ModelCheckpoint('best_wafer_SACNN_29class_model.keras', save_best_only=True, monitor='val_loss', mode='min') 
]

start_time = time.time()
history = model.fit(
    X_train, y_train_int, 
    epochs=100,
    batch_size=64,
    validation_data=(X_val, y_val_int),
    callbacks=callbacks,
    verbose=1
)
end_time = time.time()
print(f"訓練總時間：{end_time - start_time:.2f} 秒")


model.save('wafer_SACNN_29class_model_final.keras')


# === 繪製訓練結果 & 混淆矩陣 ===
print("\n正在繪製訓練歷史和混淆矩陣...")
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_history_29class.png")
print("訓練歷史圖已儲存為 training_history_29class.png")

print("\n正在載入最佳模型以繪製混淆矩陣...")
try:
    best_model = tf.keras.models.load_model('best_wafer_SACNN_29class_model.keras')
except Exception as e:
    print(f"載入 'best_wafer_SACNN_29class_model.keras' 失敗: {e}")
    best_model = model

print("正在對測試集進行預測...")
y_pred_probs = best_model.predict(X_test)
y_pred_int = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test_int, y_pred_int)
plt.figure(figsize=(20, 18))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix (29 Classes) - Test Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig("confusion_matrix_29class.png")
print("混淆矩陣已儲存為 confusion_matrix_29class.png")



print("\n" + "="*30)
print(" 正在載入「最佳 SACNN 模型」進行最終評估... ")
print("="*30)
print("\n--- 正在評估「測試集」(Test Set) ---")
test_results = best_model.evaluate(X_test, y_test_int, verbose=1)

print("\n--- 最終測試結果 (Test Set) ---")
print(f"  測試集 Loss:            {test_results[0]:.4f}")
print(f"  測試集 Accuracy:        {test_results[1]:.4f}")
print("="*38)